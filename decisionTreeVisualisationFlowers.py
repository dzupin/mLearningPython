import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot



#package scikit-learn provides many sample datasets
flowers = load_iris()
print flowers.feature_names
print flowers.target_names

#print values of first entry features/data as well as target a.k.a flower names (e.g. 0=setosa 1=versicolor 2=virginica)
print flowers.data[0]
print flowers.target[0]

#Print out all entries
for i in range(len(flowers.target)):
    print "Flower %d: label index:%s features: %s" % (i, flowers.target[i], flowers.data[i])

#Split up data to training and testing sets
test_idx=[0,50,100]
#training data (remove 3 entries from your data)
train_target = np.delete(flowers.target, test_idx)
train_data = np.delete(flowers.data, test_idx,axis=0)

#testing data (contains only 3 previously removed examples)
test_target = flowers.target[test_idx]
test_data = flowers.data[test_idx]

#Create decission tree classifier
classifier = tree.DecisionTreeClassifier()

#Train your currently empty classifier
classifier.fit(train_data,train_target)

#Print labels of originally removed data, followed up by prediction for those particular samples (they should match)
print test_target
print classifier.predict(test_data)

#Visualize your classifier  (Pydot currently works only for Python 2.7 and also needs:  sudo apt-get install graphviz )
dot_data = StringIO()
tree.export_graphviz(classifier,out_file=dot_data,feature_names=flowers.feature_names, class_names=flowers.target_names,
                     filled=True, rounded=True,impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("flowers.pdf")


#Print original values of your 3 test samples (you can use pdf file with decision tree algorithm to see if your model is OK)
print flowers.feature_names, flowers.target_names
print test_data[0], test_target[0]
print test_data[1], test_target[1]
print test_data[2], test_target[2]

