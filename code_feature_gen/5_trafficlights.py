# Imports
import numpy as np
import pandas as pd
import os
import time
t1 = time.time()

# ___________________________________________________________________________________________
# Traffic lights

# Load roads
print "Loading roads data..."
df_roads = pd.read_csv("../data_provided/roads.csv", usecols=[0,1,2,3,5,6,8,33,34])

# Load traffic lights
# download traffic_lights.csv from:
# http://vicroadsopendata.vicroadsmaps.opendata.arcgis.com/datasets/1f3cb954526b471596dbffa30e56bb32_0
print "Loading traffic light data..."
df_lights = pd.read_csv("../data_external/traffic_lights.csv")
df_lights.rename(columns={'\xef\xbb\xbfX': 'X'}, inplace=True)

# Haversine formula for distance
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

# Loop through each road and record nearest traffic light + some other stuff
print "Producing traffic light features..."
road_light_dist, road_light_count001, road_light_count010, road_light_count100, road_light_count200 = [], [], [] ,[], []
for i in range(len(df_roads)):
    rlat, rlon = df_roads["Latitude"][i], df_roads["Longitude"][i]
    km = haversine_np(df_lights['X'],df_lights['Y'],rlon,rlat)
    road_light_dist.append(km.min())
    road_light_count001.append(len(km[km<0.1]))
    road_light_count010.append(len(km[km<1.0]))
    road_light_count100.append(len(km[km<10.0]))
    road_light_count200.append(len(km[km<20.0]))

print "Saving traffic light features..."
df_roads["G_TLIGHT_DIST"] = road_light_dist
df_roads["G_TLIGHT_100m"] = road_light_count001
df_roads["G_TLIGHT_1km"]  = road_light_count010
df_roads["G_TLIGHT_10km"] = road_light_count100
df_roads["G_TLIGHT_20km"] = road_light_count200
df_roads[["ROAD_ID",
          "G_TLIGHT_DIST",
          "G_TLIGHT_100m",
          "G_TLIGHT_1km",
          "G_TLIGHT_10km",
          "G_TLIGHT_20km"]].to_csv("../features/traffic_lights-roadid.txt", sep="\t", index=False)


print ">>> Traffic light features done in", round((time.time() - t1)/60.,2),"minutes"