import pandas as pd
import sklearn as sk
from sklearn import tree

df = pd.read_csv('Hockey.csv')
X = df.iloc[:, 0:3].values
Y = df.iloc[:, -1].values

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf.predict([[5, 90, 26]]))
