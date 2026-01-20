import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

df = pd.read_csv("joint_data_collection.csv")  
# columns: state, year, population, employment, gdp

model = load_model("currentAiSolution.h5", compile=False)


output_dir = "predictions/gdp_forecasts"
os.makedirs(output_dir, exist_ok=True)

def project_variable(years, values, future_years):
    X = years.reshape(-1, 1)
    y = values

    model = LinearRegression()
    model.fit(X, y)

    last_year = years.max()
    future_X = np.array(
        [last_year + i for i in range(1, future_years + 1)]
    ).reshape(-1, 1)

    return future_X.flatten(), model.predict(future_X)


future_horizon = 5
features = ["population", "employment", "year"]

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

for state in df["state"].unique():
    state_df = df[df["state"] == state].sort_values("year")

    years = state_df["year"].values
    population = state_df["population"].values
    employment = state_df["employment"].values

    # ---- Project population & employment ----
    future_years, pop_future = project_variable(years, population, future_horizon)
    _, emp_future = project_variable(years, employment, future_horizon)

    gdp_preds = []

    for y, p, e in zip(future_years, pop_future, emp_future):
        X_future = np.array([[p, e, y]])
        X_future = scaler.transform(X_future)
        gdp = model.predict(X_future, verbose=0)[0][0]
        gdp_preds.append(gdp)


plt.figure(figsize=(7, 4))
plt.plot(future_years, gdp_preds, marker="o", label="Predicted GDP")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.title(f"GDP Forecast for {state}")
plt.grid(True)
plt.legend()

plt.savefig(f"{output_dir}/{state}.png", dpi=300)
plt.close()

