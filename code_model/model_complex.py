import numpy as np
import pandas as pd
import os
from collections import OrderedDict
import xgboost as xgb
import scipy
import random
import time
import sklearn.metrics

# __________________________________________________________________
# Load data
t1 = time.time()
# Load the train/test
print "Loading train/test data..."
df_train    = pd.read_csv("../features/train1.txt", sep="\t")
df_test     = pd.read_csv("../features/test1.txt", sep="\t")

# Load features
print "Loading features..."
df_roads    = pd.read_csv("../features/roads-roadid.txt", sep="\t")
df_wx       = pd.read_csv("../features/precipitation-stationid_year_qtr.txt", sep="\t")
df_seasonal = pd.read_csv("../features/seasonal-year_qtr.txt", sep="\t")
df_popn     = pd.read_csv("../data_external/population.txt", sep="\t")
df_lights   = pd.read_csv("../features/traffic_lights-roadid.txt", sep="\t")
df_cities   = pd.read_csv("../features/cities-roadid.txt", sep="\t")

# Load mappings
print "Loading mappings..."
df_roadwx   = pd.read_csv("../features/map_roadid_stationid.txt", sep="\t")
df_roadlga  = pd.read_csv("../features/map_roadid_lga.txt", sep="\t")
print ">>> Data loaded in", round((time.time() - t1)/60.,2),"minutes"


# __________________________________________________________________
# Function to join all the features on
def join_features(df_in):

    # ------------------------- Join on the features
    # Join blocks (required for CV)
    df_in = pd.merge(df_in, df_roads[["ROAD_ID","BLOCK"]], on="ROAD_ID", how="left")
    
    # Main road features
    df_in = pd.merge(df_in, df_roads.drop("BLOCK", axis=1), on="ROAD_ID", how="left")
    df_in.drop(["ROAD_NAME_MAP","DISTANCE"], axis=1, inplace=True)
    
    # Weather
    df_in = pd.merge(df_in, df_roadwx, on="ROAD_ID", how="left")
    df_in = pd.merge(df_in, df_wx, on=["station","year","qtr"])
    df_in.drop(["station"], axis=1, inplace=True)
    
    # Seasonal
    df_in = pd.merge(df_in, df_seasonal, on=["year","qtr"], how="left")

    # Population
    df_in = pd.merge(df_in, df_roadlga, on="ROAD_ID", how="left")
    df_in = pd.merge(df_in, df_popn, on=["LGA","year"], how="left")
    df_in.drop(["LGA"], axis=1, inplace=True)

    # Traffic lights
    df_in = pd.merge(df_in, df_lights, on="ROAD_ID", how="left")
    
    # Nearest city
    df_in = pd.merge(df_in, df_cities, on="ROAD_ID", how="left")

    # ------------------------- Clean up
    # Remove train/test level columns which are no longer needed
    df_in.drop(['ROAD_ID', 'QUARTER', 'qtr', 'year'], axis=1, inplace=True)
    
    # Return
    return df_in



# __________________________________________________________________
# Do the joins
t1 = time.time()

# Get stuff for regression model
df_train_reg = df_train[df_train["COST"]>0]
df_train_reg = join_features(df_train_reg) 
df_train_reg["COST"] = df_train_reg["COST"].clip(upper=330)

# Get stuff for classification model
df_train_nonzero = df_train[df_train["COST"]>0]
def make_class_dataset():
    df_train_zero = df_train[df_train["COST"]==0].sample(n=9*len(df_train_nonzero))
    df_train_classx = pd.concat([df_train_nonzero, df_train_zero])
    df_train_classx = join_features(df_train_classx)
    df_train_classx["label"] = (df_train_classx["COST"]>0).astype(np.int)
    return df_train_classx
df_train_class = []
for i in range(4):
    df_train_class.append(make_class_dataset())

# Get stuff for test set
df_test1 = join_features(df_test)

print ">>> Features joined in", round((time.time() - t1)/60.,2),"minutes"



# __________________________________________________________________
# General function to do block-based CV
def build_block_cv(df_input, col_exclude, col_response, eval_metric, params):
    N_CV = 5
    # Split by blocks
    seed = 0.5
    blocks = df_input["BLOCK"].unique()
    random.shuffle(blocks, lambda: seed)
    blocks = blocks[0:len(blocks)-(len(blocks)%N_CV)]
    df_input = df_input[df_input["BLOCK"].isin(blocks)].reset_index()
    cv_blocks = np.split(blocks, N_CV)
    
    # Can we use xgb cv
    train_idx, test_idx = [], []
    for cv_split in cv_blocks:
        train_idx.append(df_input[~df_input["BLOCK"].isin(cv_split)].index.tolist())
        test_idx.append(df_input[df_input["BLOCK"].isin(cv_split)].index.tolist())
    
    # Fixup the train one to make it XGB friendly
    col_exclude = col_exclude + [col_response] + ["BLOCK","index"]
    trainresponse_data = df_input[col_response].values
    trainfeatures_data = df_input.drop(col_exclude, axis=1)
    trainfeature_cols = trainfeatures_data.columns.tolist()

    # Make the XGB matricies
    xgb_train = xgb.DMatrix(trainfeatures_data, label=trainresponse_data, feature_names=trainfeature_cols)

    # Build CV model
    model_cv = xgb.cv(params,\
                      xgb_train,\
                      num_boost_round = 2000,\
                      nfold = N_CV,\
                      early_stopping_rounds=10,\
                      verbose_eval=False,\
                      folds=zip(train_idx,test_idx))

    print "Number of trees:",len(model_cv)
    err = model_cv["test-" + eval_metric + "-mean"][-1:]
    print err

    x_trees = np.asarray(range(len(model_cv)))+1
    
    # Build the model based on CV
    watchlist = [(xgb_train, 'train')]
    num_rounds = len(model_cv)
    xgb_results = {}
    model = xgb.train(params,\
                      xgb_train,\
                      num_rounds,\
                      evals=watchlist,\
                      evals_result=xgb_results,\
                      verbose_eval=False)
    
    return model


# __________________________________________________________________
# Build the regression (severity) model
t1 = time.time()
exclude_cols_reg = ["ID"]
response_reg = "COST"

# Define parameters
eval_metric_reg = "rmse"
params_reg = {
        # Booster parameters
        'eta'              : 0.01,
        'min_child_weight' : 10,
        'max_depth'        : 3,
        'gamma'            : 0.1,
        'subsample'        : 0.8,
        'colsample_bytree' : 1,
        'lambda'           : 0,
        'alpha'            : 0,
        # Learning task
        'objective'        : "reg:linear",
        'eval_metric'      : eval_metric_reg
       }
        
model_cv_reg1 = build_block_cv(df_train_reg, exclude_cols_reg, response_reg, eval_metric_reg, params_reg)
print ">>> Regression (severity) model built in", round((time.time() - t1)/60.,2),"minutes"


# __________________________________________________________________
# Build the classification (frequency) model
t1 = time.time()
exclude_cols_class = ["ID","COST"]
response_class = "label"

# Define parameters
eval_metric_class = "logloss"
params_class = {
        # Booster parameters
        'eta'              : 0.1,
        'min_child_weight' : 2,
        'max_depth'        : 5,
        'gamma'            : 1,
        'subsample'        : 0.8,
        'colsample_bytree' : 1,
        'lambda'           : 1,
        'alpha'            : 0,
        # Learning task
        'objective'        : "binary:logistic",
        'eval_metric'      : eval_metric_class
       }

# Make four classifications, with different sets.
# This is because the 0 responses are chosen at random, and this way we utilise more data
model_cv_class1 = build_block_cv(df_train_class[0], exclude_cols_class, response_class, eval_metric_class, params_class)
model_cv_class2 = build_block_cv(df_train_class[1], exclude_cols_class, response_class, eval_metric_class, params_class)
model_cv_class3 = build_block_cv(df_train_class[2], exclude_cols_class, response_class, eval_metric_class, params_class)
model_cv_class4 = build_block_cv(df_train_class[3], exclude_cols_class, response_class, eval_metric_class, params_class)
print ">>> Classification (frequency) models (x4) built in", round((time.time() - t1)/60.,2),"minutes"


# __________________________________________________________________
# Score
# Setup test dataframe for XGB matrix
t1 = time.time()
exclude_cols_test = ["ID","BLOCK"] #+ direction_cols
testfeatures_data = df_test1.drop(exclude_cols_test, axis=1)
testfeature_cols = testfeatures_data.columns.tolist()
xgb_test = xgb.DMatrix(testfeatures_data, feature_names=testfeature_cols)

# Make predictions

df_test2 = df_test1.copy()
df_test2["freq"] = (model_cv_class1.predict(xgb_test) +\
                   model_cv_class2.predict(xgb_test) +\
                   model_cv_class3.predict(xgb_test) +\
                   model_cv_class4.predict(xgb_test))/4.0

df_test2["sev"] = model_cv_reg1.predict(xgb_test)


# Unwind the frequency predictions (to compensate for up-sampling)
t = (df_train["COST"] > 0).astype(np.int).mean() # rate of 1s in population
y = (df_train_class[0]["COST"] > 0).astype(np.int).mean() # rate of 1s in model
unwind = 1/(1+(1/t-1)/(1/y-1)*(1/df_test2["freq"]-1))
df_test2["freq_unwind"] = unwind

# Combine
df_test2["COST"] = df_test2["freq_unwind"] * df_test2["sev"]
df_test2[["ID","COST"]].to_csv("predictions.csv", index=False)
print ">>> Scored and output predictions in", round((time.time() - t1)/60.,2),"minutes"
