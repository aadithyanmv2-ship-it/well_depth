
import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

data = pd.read_csv("well.csv")

X = data[["rainfall", "soil_type", "water_level"]]
y = data["well_depth"]

preprocessor = ColumnTransformer([
    ("soil", OneHotEncoder(handle_unknown="ignore"), ["soil_type"]),
    ("num", StandardScaler(), ["rainfall", "water_level"])
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X, y)

print("Model trained successfully!")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    location = None
    error = None

    if request.method == "POST":
        try:
            location = request.form["location"].strip()
            rainfall = float(request.form["rainfall"])
            soil_type = request.form["soil_type"].strip().lower()
            water_level = float(request.form["water_level"])

            if not location:
                raise ValueError("Please enter a village name.")

            if rainfall < 0 or water_level < 0:
                raise ValueError("Values cannot be negative.")

            input_data = pd.DataFrame({
                "rainfall": [rainfall],
                "soil_type": [soil_type],
                "water_level": [water_level]
            })

            prediction = round(model.predict(input_data)[0], 2)

        except ValueError as e:
            error = str(e)

        except Exception:
            error = "Unable to make prediction. Check your input."

    return render_template(
        "index.html",
        prediction=prediction,
        location=location,
        error=error
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
