import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df[["population", "employment", "year"]]
y_train = train_df["gdp"]

X_test = test_df[["population", "employment", "year"]]
y_test = test_df["gdp"]

model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1)  # output layer (GDP)
])

print(model.summary())

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)

from sklearn.metrics import r2_score

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

rmse = np.sqrt(test_loss)
print("RMSE:", rmse)

from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", mape * 100, "%")

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.xlabel("Actual GDP")
plt.ylabel("Predicted GDP")
plt.title("Actual vs Predicted GDP")
plt.legend()
plt.tight_layout()
plt.savefig("Actual vs Predicted GDP.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("Loss Curve.png")
plt.close()


# model.save("currentAiSolution")

# XML-like alternative (if strictly required)
model.save("currentAiSolution.h5")






import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_vs_validation_loss.png")
plt.close()

plt.figure()
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.title("Training vs Validation MAE")
plt.legend()
plt.tight_layout()
plt.savefig("training_vs_validation_mae.png")
plt.close()

y_pred = model.predict(X_test)

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual GDP")
plt.ylabel("Predicted GDP")
plt.title("Actual vs Predicted GDP")
plt.tight_layout()
plt.savefig("actual_vs_predicted_gdp.png")
plt.close()

residuals = y_test - y_pred.flatten()

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted GDP")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted GDP")
plt.tight_layout()
plt.savefig("residuals_plot.png")
plt.close()






import pandas as pd
import numpy as np

train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df[["population", "employment", "year"]]
y_train = train_df["gdp"]

X_test = test_df[["population", "employment", "year"]]
y_test = test_df["gdp"]

import statsmodels.api as sm

X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test)

ols_model = sm.OLS(y_train, X_train_ols).fit()

y_pred_ols = ols_model.predict(X_test_ols)

with open("currentOlsSolution.txt", "w") as f:
    f.write(ols_model.summary().as_text())

from sklearn.metrics import mean_squared_error, mean_absolute_error

ols_mse = mean_squared_error(y_test, y_pred_ols)
ols_mae = mean_absolute_error(y_test, y_pred_ols)

print("OLS Test MSE:", ols_mse)
print("OLS Test MAE:", ols_mae)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

residuals = y_test - y_pred_ols

plt.figure()
plt.scatter(y_pred_ols, residuals, color="blue", alpha=0.7, label="Residuals")
plt.axhline(0, color="red", linestyle="--", label="Zero Error Line")
plt.xlabel("Predicted GDP")
plt.ylabel("Residuals")
plt.title("OLS Residuals vs Predicted GDP")
plt.legend()
plt.tight_layout()
plt.savefig("ols_residuals_plot.png")
plt.close()

sm.qqplot(residuals, line="45", marker="o", color="green")
plt.title("OLS Q–Q Plot (Residual Normality)")
plt.tight_layout()
plt.savefig("ols_qq_plot.png")
plt.close()


plt.figure()
plt.scatter(y_test, y_pred_ols, color="purple", alpha=0.7, label="Predicted vs Actual")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="black",
    linestyle="--",
    label="Perfect Prediction"
)
plt.xlabel("Actual GDP")
plt.ylabel("Predicted GDP")
plt.title("OLS: Actual vs Predicted GDP")
plt.legend()
plt.tight_layout()
plt.savefig("ols_actual_vs_predicted.png")
plt.close()


ann_mse = test_loss   # from ANN
ann_mae = test_mae

comparison_df = pd.DataFrame({
    "Model": ["ANN", "OLS"],
    "MSE": [ann_mse, ols_mse],
    "MAE": [ann_mae, ols_mae]
})

print(comparison_df)
comparison_df.to_csv("model_comparison.csv", index=False)

