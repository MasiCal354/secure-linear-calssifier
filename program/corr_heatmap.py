import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Reading input dataset")
df = pd.read_csv("../inputs/preprocessed.csv")
print("Computing correlation matrix")
corr_matrix = df.corr().round(2)

print("Plotting correlation matrix heatmap")
plt.figure(figsize=(16,16))
sns.heatmap(corr_matrix, annot=True)
plt.show()