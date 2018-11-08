from sklearn import datasets
#from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import sys
import matplotlib
if sys.version_info[0] > 2:
    matplotlib.use('Qt5Agg') # You need to installpyqt5 library for Python3: sudo apt-get install python3-pyqt5
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
predicted = cross_val_predict(lr, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()