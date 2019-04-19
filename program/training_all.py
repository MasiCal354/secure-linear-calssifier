import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import pickle

print("Read input dataset")
df = pd.read_csv("../inputs/preprocessed.csv")

print("Split Dataset")
X = np.array(df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]])
y = np.array(df["diagnosis"])
X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Training...")
clf = SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)

if(clf.score(X_test,y_test) > 0.931):
	print("Saving Model and dataset split")
	with open("../inputs/model_all.pickle", "wb") as f:
		pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
	with open("../inputs/X_train_all.pickle", "wb") as f:
		pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
	with open("../inputs/X_test_all.pickle", "wb") as f:
		pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)
	with open("../inputs/y_train_all.pickle", "wb") as f:
		pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
	with open("../inputs/y_test_all.pickle", "wb") as f:
		pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)

print("Done!")