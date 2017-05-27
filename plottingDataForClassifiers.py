import numpy as np
import matplotlib
import sys
if sys.version_info[0] > 2:
    matplotlib.use('Qt5Agg') # You need to installpyqt5 library for Python3: sudo apt-get install python3-pyqt5
import matplotlib.pyplot as plt

#to install matplotlib module (sudo pip install matplotlib) you first need to run: sudo apt install python-gtk2-dev

greyhounds = 500
labradors = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labradors)

#Print your plots
plt.hist([grey_height,lab_height],stacked=True, color=['r','b'])
plt.show()