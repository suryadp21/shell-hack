import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load your dataset, replace '<DATA_PATH>' with the actual file path
data = pd.read_csv('/content/Biomass_History.csv')

# Extract features and target variable for the first 500 rows
X = data[['2010', '2011', '2012', '2013', '2014', '2015', '2016']].iloc[:500]
y = data['2017'].iloc[:500]
#for 500 values
# Create XGBoost regressor
xgb_regressor = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the model
xgb_regressor.fit(X, y)

# Predict on the same data for demonstration purposes
y_pred = xgb_regressor.predict(X)

# Calculate Mean Squared Error on the training data
mse = mean_squared_error(y, y_pred)

print(f"Mean Squared Error: {mse}")

# Add predicted 2017 values and difference columns to the DataFrame
data_subset = data.iloc[:500]  # Select the same subset of data used for modeling
data_subset['Predicted_2017'] = y_pred
data_subset['Difference'] = data_subset['2017'] - y_pred

# Display the modified DataFrame with added columns
print(data_subset)
