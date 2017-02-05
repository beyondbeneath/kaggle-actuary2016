#!/bin/sh

# ------------------
# FEATURE GENERATION
# ------------------

python 1_roads.py         # ~ 1 minute
python 2_weather.py       # ~ 8 minutes
python 3_seasonal.py      # ~ a few seconds
python 4_population.py
python 5_trafficlights.py # ~ 10 minutes
python 6_cities.py        # a few seconds