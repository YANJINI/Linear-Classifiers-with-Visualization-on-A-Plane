import sys
sys.path.extend(['/Users/jinijani/PycharmProjects/git_project/ML/Linear-Classifiers-with-Visualization-on-A-Plane'])

from Perceptron import twoD_Perceptron
from linear_GNB import twoD_lGNB
from logistic_r import twoD_logisticR
from softmargin_SVM import twoD_softmarginSVM

a = twoD_Perceptron(n_iters= 200)
b = twoD_lGNB()
c = twoD_logisticR()
d = twoD_softmarginSVM(C=10)
d.with_different_C(C=0.0001)