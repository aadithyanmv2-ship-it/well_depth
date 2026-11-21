import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("well.csv")

location_encoder = LabelEncoder()
soil_encoder = LabelEncoder()

data['Location'] = location_encoder.fit_transform(data['Location'])
data['Soil_Type'] = soil_encoder.fit_transform(data['Soil_Type'])

X = data[['Location', 'Rainfall', 'Soil_Type', 'Water_Level']]
y = data['Well_Depth']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

location = input("Enter Location: ")
rainfall = float(input("Enter Rainfall (mm): "))
soil = input("Enter Soil Type: ")
water = float(input("Enter Water Level (meters): "))

location_encoded = location_encoder.transform([location])[0]
soil_encoded = soil_encoder.transform([soil])[0]

result = model.predict([[
    location_encoded,
    rainfall,
    soil_encoded,
    water
]])

print("Predicted Well Depth:", round(result[0], 2), "meters")