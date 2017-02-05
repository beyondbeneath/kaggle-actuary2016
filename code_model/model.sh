#!/bin/sh

# ------------------------------------
# RUN A COMPLEX AND SIMPLE MODEL
# ------------------------------------

# Run the complex model (1 minute load, 5 minutes training/score)
# ~ 1 minute to load/join data
# ~ 15 seconds to run regression (severity) model
# ~ 4.5 minutes to run 4x classification (frequency) models
# ~ 20 seconds to score/save predictions
python model_complex.py

# Run the complex model (<1 minute load, <1 minute training/score)
# ~ 20 seconds to load/join data
# ~ 10 seconds to run regression (severity) model
# ~ 30 seconds to run classification (frequency) model
# ~ 10 seconds to score/save predictions
python model_simple.py