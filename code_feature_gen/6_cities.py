# Imports
import numpy as np
import pandas as pd
import os
import time
t1 = time.time()

# ___________________________________________________________________________________________
# Dist to nearest cities
df_roads = pd.read_csv("../data_provided/roads.csv", usecols=[0,5,6,34])

# Define cities
cities = {}
cities['geelong']     = [-38.1499, 144.3617]
cities['ballarat']    = [-37.5622, 143.8503]
cities['bendigo']     = [-36.7570, 144.2794]
cities['melton']      = [-37.6691, 144.5437]
cities['shepparton']  = [-36.3833, 145.4000]
cities['mildura']     = [-34.2080, 142.1246]
cities['wodonga']     = [-36.1241, 146.8818]
cities['warrnambool'] = [-38.3687, 142.4982]

# Haversine for distances
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

# Find distance to city for each road
distances = {}
for city, coords in cities.iteritems():
    km = haversine_np(coords[1],coords[0],df_roads["Longitude"],df_roads["Latitude"])
    df_roads["dist_to_"+city] = km
    
df_roads.drop(["Latitude","Longitude","BLOCK"], axis=1, inplace=True)
df_roads.to_csv("../features/cities-roadid.txt", sep="\t", index=False)

print ">>> City distance features done in", round((time.time() - t1)/60.,2),"minutes"