
import os
import math
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset using an absolute path
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "well.csv"

# Check dataset
if not DATA_PATH.exists():
    raise FileNotFoundError("well.csv file not found!")

data = pd.read_csv(DATA_PATH)

# Clean dataset
data["soil_type"] = data["soil_type"].astype(str).str.strip().str.lower()
data = data.dropna(subset=[
    "rainfall", "soil_type", "water_level", "well_depth"
])

# Input and output
X = data[["rainfall", "soil_type", "water_level"]]
y = data["well_depth"]

# Preprocessing
preprocessor = ColumnTransformer([
    (
        "soil",
        OneHotEncoder(handle_unknown="ignore"),
        ["soil_type"]
    ),
    (
        "num",
        StandardScaler(),
        ["rainfall", "water_level"]
    )
])

# Machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train model
model.fit(X, y)

print("Model trained successfully!")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    location = None
    error = None

    if request.method == "POST":
        try:
            location = request.form.get("location", "").strip()

            rainfall = float(request.form.get("rainfall", ""))
            soil_type = request.form.get("soil_type", "").strip().lower()
            water_level = float(request.form.get("water_level", ""))

            if not location:
                raise ValueError("Please enter a village name.")

            if not soil_type:
                raise ValueError("Please select a soil type.")

            if not math.isfinite(rainfall) or not math.isfinite(water_level):
                raise ValueError("Please enter valid numeric values.")

            if rainfall < 0 or water_level < 0:
                raise ValueError("Values cannot be negative.")

            input_data = pd.DataFrame({
                "rainfall": [rainfall],
                "soil_type": [soil_type],
                "water_level": [water_level]
            })

            prediction = round(float(model.predict(input_data)[0]), 2)

        except ValueError as e:
            error = str(e)

        except Exception:
            error = "Unable to make prediction. Please check your input."

    return render_template(
        "index.html",
        prediction=prediction,
        location=location,
        error=error
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
