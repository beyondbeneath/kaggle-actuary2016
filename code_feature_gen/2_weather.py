# Imports
import numpy as np
import pandas as pd
import os
import time
import glob
t1 = time.time()


# ___________________________________________________________________________________________
# Produce a road_id -> station_id mapping
print "Mapping road_id to station_id..."
# Get roads
df_roads = pd.read_csv("../data_provided/roads.csv", usecols=[0,5,6])
df_roads.columns = ["ROAD_ID","lat","lon"]

# Get stations
df_stn = pd.read_csv(os.path.join(os.getcwd(),"../data_external/weather/DC02D_StnDet_999999999324140.txt"), usecols=[1,6,7])
df_stn.columns = ['station','lat','lon']

# Haversine distance
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

# Loop through each road and record nearest wx station
road_stn = []
for i in range(len(df_roads)):
    rlat, rlon = df_roads["lat"][i], df_roads["lon"][i]
    df_stn_tmp = df_stn.copy()
    km = haversine_np(df_stn['lon'],df_stn['lat'],rlon,rlat)
    df_stn_tmp["dist"] = km
    idx_max = df_stn_tmp["dist"].argmin()
    road_stn.append(int(df_stn_tmp.ix[idx_max]["station"]))

# Attach it on as a column, and save it to a new lookup table
df_roads["station"] = road_stn
df_roads[["ROAD_ID","station"]].to_csv("../features/map_roadid_stationid.txt", sep="\t", index=False)


# ___________________________________________________________________________________________
# Load all the wx data and produce a file of average precipition per year per quarter per station
print "Loading wx data..."
wx_files = glob.glob(os.path.join(os.getcwd(),"../data_external/weather/DC02D_Data*.*"))
df_wx = None
for w in wx_files:
    if type(df_wx) == None:
        df_wx = pd.read_csv(w, usecols=[1,2,3])
    else:
        df_wx = pd.concat([df_wx, pd.read_csv(w, usecols=[1,2,3])])
        
df_wx.columns = ['station', 'date','precip']

# Remove white space
df_wx["precip"] = df_wx["precip"].str.replace(" ","")

# Map to float
def mapwx(x):
    if x == "": return 0
    else:
        try:
            return float(x)
        except:
            print x              
df_wx["precip"] = df_wx["precip"].map(mapwx)

# Add year,qtr
def wxyear(x):
    return int(x[-4:])

def wxqtr(x):
    mth = int(x[3:5])
    if mth <= 3: return 1
    elif mth <=6: return 2
    elif mth <= 9: return 3
    elif mth <= 12: return 4
    
df_wx["year"] = df_wx["date"].map(wxyear)
df_wx["qtr"] = df_wx["date"].map(wxqtr)

print "Aggregating and saving wx data..."
df_wxg = df_wx.groupby(["station", "year","qtr"], as_index=False).agg({"precip":np.mean})
df_wxg.to_csv("../features/precipitation-stationid_year_qtr.txt", index=False, sep="\t")

print ">>> Wx features done in", round((time.time() - t1)/60.,2),"minutes"

