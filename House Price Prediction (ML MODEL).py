# Simple house price prediction project
# Using Decision Tree and Random Forest

# 1. Import libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# 2. Load data

melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# remove rows without price
melbourne_data = melbourne_data.dropna(axis=0)


# 3. Select target and features

y = melbourne_data.Price  # target variable
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[features]  # features for the model


# 4. Split data into train and validation

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


# 5. Decision Tree model

dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(train_X, train_y)  # train the model

# predict on validation data
dt_preds = dt_model.predict(val_X)

# calculate error
dt_mae = mean_absolute_error(val_y, dt_preds)
print("Decision Tree MAE:", dt_mae)


# 6. Random Forest model

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)  # train the model

# predict on validation data
rf_preds = rf_model.predict(val_X)

# calculate error
rf_mae = mean_absolute_error(val_y, rf_preds)
print("Random Forest MAE:", rf_mae)


# 7. Compare models

if dt_mae < rf_mae:
    print("Decision Tree is better on validation data")
else:
    print("Random Forest is better on validation data")

