from sklearn import tree

#Training data: features[weight in grams, surface bumpy=1/smooth=0], labels[ 0=Orange and 4=Apple]
features =[ [140, 1], [130, 1], [150,0], [170,0]]
labels = [ 0, 0, 4, 4]

# Classifier - Box of rules
classifier = tree.DecisionTreeClassifier()

# Fill your currently empty classifier with set of rules using learning algorithm: fit (find patterns in the data)
classifier = classifier.fit(features,labels)

#use your trainned classifier to predict/classify new samples
#Comment: Result bellow is [0]: and it shows that heavy fruit with bumpy surface is predicted to be Orange
print classifier.predict([[150,1]])

#Comment: Result below is [4]: and it shows that heavy fruit with smooth surface is predicted to be Apple.
print classifier.predict([[190,0]])
