import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from phe import paillier
from tqdm import tqdm
import pickle

print("Generating Keypair")
public_key, private_key = paillier.generate_paillier_keypair()

print("Loading model")
with open("../inputs/model_two_att.pickle", "rb") as f:
	clf = pickle.load(f)

print("Loading Testing Data")
with open("../inputs/X_test_two_att.pickle", "rb") as f:
	X_test = pickle.load(f)
with open("../inputs/y_test_two_att.pickle", "rb") as f:
	y_test = pickle.load(f)

print("Encrypting Test Data")
enc_X_test = [[public_key.encrypt(j) for j in i] for i in tqdm(X_test)]
enc_X_test = np.array(enc_X_test)

print("Mapping encrypted testing data")
enc_mapping = list()
for i in tqdm(enc_X_test):
    tot = 0
    for j in range(len(i)):
        tot += i[j] * clf.coef_[0,j]
    enc_mapping.append(tot)

print("Decrypt mapping value")
dec_mapping = [private_key.decrypt(i) for i in tqdm(enc_mapping)]

print("Get prediction")
y_pred = [0 if i < -clf.intercept_[0] else 1 for i in dec_mapping]
y_pred = np.array(y_pred)

print("Evaluate Prediction")
print("Accuracy Score: ", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Done!")