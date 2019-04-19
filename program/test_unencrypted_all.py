import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Loading model")
with open("../inputs/model_all.pickle", "rb") as f:
	clf = pickle.load(f)

print("Loading Testing Data")
with open("../inputs/X_test_all.pickle", "rb") as f:
	X_test = pickle.load(f)
with open("../inputs/y_test_all.pickle", "rb") as f:
	y_test = pickle.load(f)

print("Get prediction")
y_pred = clf.predict(X_test)

print("Evaluate Prediction")
print("Accuracy Score: ", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Done!")