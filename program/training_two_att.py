import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import pickle

print("Read input dataset")
df = pd.read_csv("../inputs/preprocessed.csv")

print("Split Dataset")
X = np.array(df[["concave points_worst", "radius_worst"]])
y = np.array(df["diagnosis"])
X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Training...")
clf = SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)

if(clf.score(X_test,y_test) > 0.85):
	print("Saving Model and dataset split")
	with open("../inputs/model_two_att.pickle", "wb") as f:
		pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
	with open("../inputs/X_train_two_att.pickle", "wb") as f:
		pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
	with open("../inputs/X_test_two_att.pickle", "wb") as f:
		pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)
	with open("../inputs/y_train_two_att.pickle", "wb") as f:
		pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
	with open("../inputs/y_test_two_att.pickle", "wb") as f:
		pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)

print("Done!")