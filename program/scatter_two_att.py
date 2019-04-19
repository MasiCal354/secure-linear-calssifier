import pandas as pd
import matplotlib.pyplot as plt

print("Reading inputs")
df = pd.read_csv("../inputs/preprocessed.csv")
print("Separate data by target")
df_label_B = df.where(df["diagnosis"] == 0)
df_label_M = df.where(df["diagnosis"] == 1)
print("Plotting the scatter plot")
plt.scatter(df_label_B["concave points_worst"], df_label_B["radius_worst"], color="blue", label="B")
plt.scatter(df_label_M["concave points_worst"], df_label_M["radius_worst"], color="red", label="M")
plt.xlabel("concave points_worst")
plt.ylabel("radius_worst")
plt.title("Map of value to be classified")
plt.legend()
plt.show()