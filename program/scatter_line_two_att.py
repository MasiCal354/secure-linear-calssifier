import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import pickle

print("Loading model")
with open("../inputs/model_two_att.pickle", "rb") as f:
	clf = pickle.load(f)
print("Read input dataset")
df = pd.read_csv("../inputs/preprocessed.csv")
X = np.array(df[["concave points_worst", "radius_worst"]])
y = np.array(df["diagnosis"])

colors = ['blue', 'red']

x_min, x_max = X[:, 0].min() - .025, X[:, 0].max() + .2
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xs = np.arange(x_min,x_max,0.5)
ys = (-clf.intercept_[0]-xs*clf.coef_[0,0])/clf.coef_[0,1]
plt.plot(xs,ys)
for i in range(len(colors)):
    px = X[:, 0][y == i]
    py = X[:, 1][y == i]
    plt.scatter(px, py, c=colors[i])
plt.xlabel("concave points_worst")
plt.ylabel("radius_worst")
plt.title("Classified value")
plt.show()