from sklearn import tree
features = [[22, 20, 5], [11, 80, 100], [36, 25, 12], [7, 75, 50], [19, 20, 5], [22, 10, 8], [50, 90, 200],[25, 30, 10],[1, 100, 90],[0, 40, 40],[20, 50, 200]]
labels = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[20, 10, 16]]))
