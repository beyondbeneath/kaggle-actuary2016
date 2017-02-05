# Imports
import numpy as np
import pandas as pd
import os
import time
import statsmodels.formula.api as smf
t1 = time.time()

# ___________________________________________________________________________________________
# Produce a simple seasonal model

# Read train/test
print "Loading train/test data..."
df_train = pd.read_csv("../features/train1.txt", sep="\t")
df_test = pd.read_csv("../features/test1.txt", sep="\t")

# Want to group by year/qtr to see seasonal pattern and longer term trend
df_train_grp = df_train.groupby(["year","qtr"]).agg({"COST":np.sum, "ROAD_ID": "count"})
df_test_grp = df_test.groupby(["year","qtr"], as_index=False).agg({"ROAD_ID": "count"})

# We'll model "COST per ROAD_ID"
df_train_grp["cost_roadid"] = df_train_grp["COST"] / df_train_grp["ROAD_ID"]

# Reset the index so can model
df_train_grp_idx = df_train_grp.reset_index()

# Build a simple linear regression model
print "Building and saving the seasonal model..."
mod = smf.ols("cost_roadid ~ 0 + year + C(qtr)", df_train_grp_idx).fit()

# Save these seasonal predictions (COST per ROAD_ID) as meta-features
# We can do it on the all the year/qtr on the test set, since the train has a subset of these
df_test_grp["seasonal"] = mod.predict(df_test_grp)
df_test_grp[["year","qtr","seasonal"]].to_csv("../features/seasonal-year_qtr.txt", sep="\t", index=False)

print ">>> Seasonal features done in", round((time.time() - t1)/60.,2),"minutes"