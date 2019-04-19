import pandas as pd

print("Reading input file...")
df = pd.read_csv("../inputs/dataset.csv")

print("Drop Unnamed: 32 column")
df = df.drop(["Unnamed: 32"], axis = 1)
print("Drop ID column")
df = df.drop(['id'], axis = 1)
print("Replace categorical target into numeric value")
df = df.replace(["B", "M"], [0, 1])

print("Save into csv")
df.to_csv("../inputs/preprocessed.csv")
print("Done!")
print("Preprocessed dataset saved into ../inputs/preprocessed.csv")