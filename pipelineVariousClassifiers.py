from sklearn import datasets
from sklearn.cross_validation import  train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

#conv
X = iris.data
y = iris.target

#divide your data into 2 sets of same size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .5)

#First version of classifier
my_classifier = tree.DecisionTreeClassifier()
#Train your classifier
my_classifier = my_classifier.fit(X_train, y_train)
#Predict using your classifier
predictions = my_classifier.predict(X_test)
print predictions
#Now check accuracy of your predictions
print accuracy_score(y_test,predictions)

#Second version of classifier
my_classifier2 = KNeighborsClassifier()
#Train your classifier
my_classifier2 = my_classifier2.fit(X_train, y_train)
#Predict using your classifier
predictions2 = my_classifier2.predict(X_test)
print predictions2
#Now check accuracy of your predictions
print accuracy_score(y_test,predictions2)



