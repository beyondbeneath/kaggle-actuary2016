# Imports
import numpy as np
import pandas as pd
import os
import re
from collections import OrderedDict
import time
t1 = time.time()

# Load original roads data
print "Loading roads data..."
df_roads = pd.read_csv("../data_provided/roads.csv")
df_roads = df_roads.fillna(0)

# _________________________________________________________________________________
# Functions to map road columns to reasonable numeric stuff
def r_postcode(x): return int(x.split(" - ")[1])

def r_landmark(x):
    strx = str(x).replace(" ","")
    find_speed2 = re.search("\d{2}km\/hr", strx)
    find_speed3 = re.search("\d{3}km\/hr", strx)
    if find_speed3 != None: speed = int(find_speed3.group(0)[0:3])
    elif find_speed2 != None: speed = int(find_speed2.group(0)[0:2])
    else: speed = -1
    return speed

def r_speed(x):
    strx = str(x).replace("km/h","").replace("mph","").replace("<","")
    return int(strx)

r_landwidth = {'Wide (? 3.25m)':3,
               'Medium (? 2.75m to < 3.25m)':2,
               'Narrow (? 0m to < 2.75m)':1}

r_paved = {'None': -1,
           'Paved >= 2.4m':3,
           'Paved 1< Width < 2.4m':2,
           'Paved 0< Width<=1m':1}

r_curve = {'Adequate':1,
           'Poor':0,
           'Not applicable':-1}

r_grade = {'0 to <4%':1,
           '7.5 to <10%':2,
           '10%+':3}

r_condition = {'Good':3,
               'Medium':2,
               'Poor':1}

r_roadside = {'>=10m':4,
              '5 to <10m':3,
              '1 to <5m':2,
              '0 to <1m':1}

# legs (?, 3, 4)
r_intersection1 = {'None':-1,
                   'Roundabout':-1,
                   '3-leg (unsignalised) with no protected turn lane':3,
                   '3-leg (unsignalised) with protected turn lane':3,
                   '3-leg (signalised) with no protected turn lane':3,
                   '3-leg (signalised) with protected turn lane':3,
                   '4-leg (signalised) with protected turn lane':4,
                   '4-leg (signalised) with no protected turn lane':4,
                   '4-leg (unsignalised) with no protected turn lane':4,
                   '4-leg (unsignalised) with protected turn lane':4,
                   'Median crossing point - formal':-1,
                   'Median crossing point - informal':-1,
                   'Merge lane':-1,
                   'Mini roundabout':-1,
                   'Railway Crossing - passive (signs only)':-1,
                   'Railway Crossing - active (flashing lights / boom gates)':-1
                  }

# Signalised
r_intersection2 = {'None':0,
                   'Roundabout':0,
                   '3-leg (unsignalised) with no protected turn lane':0,
                   '3-leg (unsignalised) with protected turn lane':0,
                   '3-leg (signalised) with no protected turn lane':1,
                   '3-leg (signalised) with protected turn lane':1,
                   '4-leg (signalised) with protected turn lane':1,
                   '4-leg (signalised) with no protected turn lane':1,
                   '4-leg (unsignalised) with no protected turn lane':0,
                   '4-leg (unsignalised) with protected turn lane':0,
                   'Median crossing point - formal':0,
                   'Median crossing point - informal':0,
                   'Merge lane':0,
                   'Mini roundabout':0,
                   'Railway Crossing - passive (signs only)':0,
                   'Railway Crossing - active (flashing lights / boom gates)':1
                  }

# Protected turn
r_intersection3 = {'None':-1,
                   'Roundabout':-1,
                   '3-leg (unsignalised) with no protected turn lane':0,
                   '3-leg (unsignalised) with protected turn lane':1,
                   '3-leg (signalised) with no protected turn lane':0,
                   '3-leg (signalised) with protected turn lane':1,
                   '4-leg (signalised) with protected turn lane':1,
                   '4-leg (signalised) with no protected turn lane':0,
                   '4-leg (unsignalised) with no protected turn lane':0,
                   '4-leg (unsignalised) with protected turn lane':1,
                   'Median crossing point - formal':-1,
                   'Median crossing point - informal':-1,
                   'Merge lane':-1,
                   'Mini roundabout':-1,
                   'Railway Crossing - passive (signs only)':-1,
                   'Railway Crossing - active (flashing lights / boom gates)':-1
                  }

# Roundabout
r_intersection4 = {'None':-1,
                   'Roundabout':1,
                   'Mini roundabout':1,
                   '3-leg (unsignalised) with no protected turn lane':0,
                   '3-leg (unsignalised) with protected turn lane':0,
                   '3-leg (signalised) with no protected turn lane':0,
                   '3-leg (signalised) with protected turn lane':0,
                   '4-leg (signalised) with protected turn lane':0,
                   '4-leg (signalised) with no protected turn lane':0,
                   '4-leg (unsignalised) with no protected turn lane':0,
                   '4-leg (unsignalised) with protected turn lane':0,
                   'Median crossing point - formal':0,
                   'Median crossing point - informal':0,
                   'Merge lane':0,
                   'Railway Crossing - passive (signs only)':0,
                   'Railway Crossing - active (flashing lights / boom gates)':0
                  }

# Railway
r_intersection5 = {'None':-1,
                   'Roundabout':0,
                   'Mini roundabout':0,
                   '3-leg (unsignalised) with no protected turn lane':0,
                   '3-leg (unsignalised) with protected turn lane':0,
                   '3-leg (signalised) with no protected turn lane':0,
                   '3-leg (signalised) with protected turn lane':0,
                   '4-leg (signalised) with protected turn lane':0,
                   '4-leg (signalised) with no protected turn lane':0,
                   '4-leg (unsignalised) with no protected turn lane':0,
                   '4-leg (unsignalised) with protected turn lane':0,
                   'Median crossing point - formal':0,
                   'Median crossing point - informal':0,
                   'Merge lane':0,
                   'Railway Crossing - passive (signs only)':1,
                   'Railway Crossing - active (flashing lights / boom gates)':1
                  }

r_intersection_q = {'Adequate':1,
                   'not applicable':-1,
                   'poor':0}

r_intersection_vol = {'Very high':5,
                      'High':4,
                      'Medium':3,
                      'Low':2,
                      'Very low':1,
                      'Not applicable':-1}

r_median = {'Physical median width >20m':21,
            'Physical median width 10-20m':20,
            'Physical median width 5-10m':10,
            'Physical median width 1-5m':5,
            'Physical median width up to 1m':1,
            'Centre line':0,
            'Central hatching (>1m)':0,
            'Wide centre line (0.3m to 1m)':0,
            'Safety barrier - metal':0,
            'Safety barrier - concrete':0,
            'Safety barrier - wire rope':0,
            'Motorcyclist friendly barrier':0,
            'Continuous central turning lane':0,
            'Oneway':0}

# skid sealed
r_skid1 = {'Sealed - adequate':1,
           'Sealed - medium':1,
           'Sealed - poor':1,
           'Unsealed - adequate':0,
           'Unsealed - poor':0}

# skid quality
r_skid2 = {'Sealed - adequate':2,
           'Sealed - medium':1,
           'Sealed - poor':0,
           'Unsealed - adequate':2,
           'Unsealed - poor':0}

r_obj_passenger = {'Non-frangible sign/ post./pole >=10cm\xa0':0,
                   'Tree >10cm\xa0':0,
                   'Frangible structure or building':0,
                   'Upwards slope - (rollover gradient)':0,
                   'None (>20m)':0,
                   'Deep drainage ditch':0,
                   'Downwards  slope':0,
                   'Non-frangible structure/bridge or building':0,
                   'Upwards steep slope (>75deg)':0,
                   'Aggressive vertical face':0,
                   'Unprotected barrier end':0,
                   'Cliff':0,
                   'Safety barrier - wire rope':1,
                   'Safety barrier - metal mc friendly':1,
                   'Safety barrier - metal':1,
                   'Safety barrier - concrete':1,
                   'Large boulders >= 20cm high':0
                  }

r_obj_driver = {'Tree >10cm\xa0':0,
                'Non-frangible sign/ post./pole >=10cm\xa0':0,
                'None (>20m)':0,
                'Frangible structure or building':0,
                'Aggressive vertical face':0,
                'Upwards slope - (rollover gradient)':0,
                'Downwards  slope':0,
                'Upwards steep slope (>75deg)':0,
                'Deep drainage ditch':0,
                'Unprotected barrier end':0,
                'Large boulders >= 20cm high':0,
                'Non-frangible structure/bridge or building':0,
                'Safety barrier - metal':1,
                'Safety barrier - concrete':1,
                'Safety barrier - wire rope':1,
                'Safety barrier - metal mc friendly':1,
                'Cliff':0
    }

# _________________________________________________________________________________
# Tell it how to map each specific column
# Mapping functions for columns
# 0-leave as is, 1-exclude, 2-dummies, 3-specific function, 4-map, 5-remap
road_col_dict = OrderedDict([
    ("ROAD_ID",[0]),
    ("ROAD_NAME",[3, r_postcode]), #Keep postcode here, in case we want to join on something
    ("CARRIAGEWAY",[2]),
    ("DISTANCE",[0]),
    ("LENGTH",[1]),
    ("Latitude",[1]),
    ("Longitude",[1]),
    ("LANDMARK",[3, r_landmark]),
    ("VEHICLE_FLOW_AADT",[0]),
    ("SPEED_LIMIT",[3, r_speed]),
    ("OPERATING_SPEED_85TH_PERCENTILE",[3, r_speed]),
    ("OPERATING_SPEED_MEAN",[3, r_speed]),
    ("LANE_WIDTH",[4, r_landwidth]),
    ("PAVED_SHOULDER_DRIVERS_SIDE",[4, r_paved]),
    ("PAVED_SHOULDER_PASSENGER_SIDE",[4, r_paved]),
    ("SHOULDER_RUMBLE_STRIPS",[2]),
    ("CURVATURE",[2]),
    ("QUALITY_OF_CURVE",[4, r_curve]),
    ("DELINEATION",[2]),
    ("GRADE",[4, r_grade]),
    ("ROAD_CONDITION",[4, r_condition]),
    ("ACCESS_POINTS",[2]),
    ("ROADSIDE_SEVERITY_PASSENGER_SIDE_DISTANCE",[4, r_roadside]),
    ("ROADSIDE_SEVERITY_PASSENGER_SIDE_OBJECT1",[5, r_obj_passenger, "ROADSIDE_SEVERITY_PASSENGER_SIDE_OBJECT", "OBJ_PASSENGER"]),
    ("ROADSIDE_SEVERITY_PASSENGER_SIDE_OBJECT",[2]),
    ("ROADSIDE_SEVERITY_DRIVERS_SIDE_DISTANCE",[4, r_roadside]),
    ("ROADSIDE_SEVERITY_DRIVERS_SIDE_OBJECT1",[5, r_obj_driver, "ROADSIDE_SEVERITY_DRIVERS_SIDE_OBJECT", "OBJ_DRIVER"]),
    ("ROADSIDE_SEVERITY_DRIVERS_SIDE_OBJECT",[2]),
    ("INTERSECTION_TYPE1",[5, r_intersection1, "INTERSECTION_TYPE", "INTERSECTION_TYPE_LEGS"]),
    ("INTERSECTION_TYPE2",[5, r_intersection2, "INTERSECTION_TYPE", "INTERSECTION_TYPE_SIGNAL"]),
    ("INTERSECTION_TYPE3",[5, r_intersection3, "INTERSECTION_TYPE", "INTERSECTION_TYPE_PROTECTED"]),
    ("INTERSECTION_TYPE4",[5, r_intersection4, "INTERSECTION_TYPE", "INTERSECTION_TYPE_ROUDNABOUT"]),
    ("INTERSECTION_TYPE5",[5, r_intersection5, "INTERSECTION_TYPE", "INTERSECTION_TYPE_RAILWAY"]),
    ("INTERSECTION_QUALITY",[4, r_intersection_q]),
    ("INTERSECTING_ROAD_VOLUME",[4, r_intersection_vol]),
    ("MEDIAN_TYPE1",[5, r_median, "MEDIAN_TYPE", "MEDIAN_TYPE_PHYSICAL"]),
    ("MEDIAN_TYPE",[2]),
    ("SKID_RESISTANCE_GRIP1",[5, r_skid1, "SKID_RESISTANCE_GRIP", "SKID_SEALED"]),
    ("SKID_RESISTANCE_GRIP2",[5, r_skid2, "SKID_RESISTANCE_GRIP", "SKID_QUALITY"]),
    ("STREET_LIGHTING",[2]),
    ("CENTRELINE_RUMBLE_STRIPS",[2]),
    ("TRAVEL_DIRECTION",[2]), # Tweak this maybe
    ("BLOCK",[0])
])

# _________________________________________________________________________________
# Map the columns, and save it
df_roads_new = df_roads.copy()
# Go through each column we want to re-map and re-map it
print "Mapping columns..."
for k in road_col_dict:
    val = road_col_dict[k]
    
    # Keep columns
    if val[0] == 0:
        x = 0
        
    # Exclude
    elif val[0] == 1:
        df_roads_new.drop(k, axis=1, inplace=True)
    
    # Dummies
    elif val[0] == 2:
        df_roads_new = pd.concat([df_roads_new, pd.get_dummies(df_roads_new[k], prefix=k)], axis=1)
        df_roads_new.drop(k, axis=1, inplace=True)
        
    # Function
    elif val[0] == 3:
        df_roads_new[k+"_MAP"] = df_roads_new[k].apply(val[1])
        df_roads_new.drop(k, axis=1, inplace=True)
        
    # Map
    elif val[0] == 4:
        df_roads_new[k+"_MAP"] = df_roads_new[k].map(val[1])
        df_roads_new.drop(k, axis=1, inplace=True)
        
    # Remap
    elif val[0] == 5:
        df_roads_new[val[3]+"_REMAP"] = df_roads_new[val[2]].map(val[1])
        
# Remove couple of the "remapped" columns and save it
print "Saving roads data..."
df_roads_new.drop(["INTERSECTION_TYPE","SKID_RESISTANCE_GRIP"], axis=1, inplace=True)
df_roads_new.columns = [x.replace("\xa0","") for x in df_roads_new.columns] # Fix the weird characters in there
df_roads_new.to_csv("../features/roads-roadid.txt", sep="\t", index=False)


# _________________________________________________________________________________
# Add some nicer columns for qtr/year to the train/test
# Load data
print "Loading train/test data..."
df_train = pd.read_csv("../data_provided/training_data.csv")
df_test = pd.read_csv("../data_provided/testing_data.csv")

# Add the qtr,year columns
print "Adding qtr/year columns..."
df_train["qtr"] = df_train["QUARTER"].map(lambda x: int(x[-1]))
df_train["year"] = df_train["QUARTER"].map(lambda x: int(x[0:4]))
df_test["qtr"] = df_test["QUARTER"].map(lambda x: int(x[-1]))
df_test["year"] = df_test["QUARTER"].map(lambda x: int(x[0:4]))

# Save
print "Saving train/test data..."
df_train.to_csv("../features/train1.txt", sep="\t", index=False)
df_test.to_csv("../features/test1.txt", sep="\t", index=False)

print ">>> Road features done in", round((time.time() - t1)/60.,2),"minutes"