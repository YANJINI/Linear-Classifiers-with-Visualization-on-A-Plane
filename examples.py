import sys
sys.path.extend(['/Users/jinijani/PycharmProjects/git_project/ML/Linear-Classifiers-with-Visualization-on-A-Plane'])
from Perceptron import twoD_coordinates_Perceptron
from linear_GNB import twoD_coordinates_lGNB
from logistic_r import twoD_coordinates_logistic_r

a = twoD_coordinates_Perceptron()
a = twoD_coordinates_lGNB()
a = twoD_coordinates_logistic_r()