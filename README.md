# RUSSIAN-PASSENGER-AIR-SERVICE-Dashboard
# Russian Passenger Air Service (2007–2020)
#  Forecasting monthly passengers per airport with Random Forest
# This notebook:
# 1. Loads and reshapes the wide-format dataset (months as columns → long format).
# 2. Builds lag features to capture temporal structure.
# 3. Trains a RandomForestRegressor to predict monthly passengers.
# 4. Evaluates model performance (MAE, RMSE).
# 5. Visualizes predictions vs actual values.
# 6. Plots feature importances and residual diagnostics.
