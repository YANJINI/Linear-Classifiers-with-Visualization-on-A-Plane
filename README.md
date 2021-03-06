# Linear-Classifiers-with-Visualization-on-A-Plane
Visualization of linear classifiers such as single Perceptron, linear GNB, logistic regression on a plane with matplotlib

## Bacic Idea
From an inspiration by [Perceptron demo](https://youtu.be/wl7gVvI-HuY?t=1331) on lecture 4, [Naive Bayes demo](https://youtu.be/rqB0XWoMreU?t=2498) on lecture 10, and [logistic regression demo](https://youtu.be/GnkDzIOxfzI?t=2703) on lecture 11 of Kilian Weinberger's [Machine Learning for Intelligent Systems course](https://www.cs.cornell.edu/courses/cs4780/2018fa/) at Cornell University, I have built conceptually same interactive single Perceptron and, linear and non-linear Gaussian naive Bayes, and logistic regression demos in Python with matplotlib event handling. <br />


## How it works
### Single Perceptron Classifier

on the figure that pops up, you click to plot from class 1 and 2 as follows. <br />

![click to plot from class 1](/images/click%20to%20plot%20from%20class%201.gif)

Press enter to switch to plotting from class 2. <br />
![click to plot from class 2](/images/click%20to%20plot%20from%20class%202.gif)

Pressing enters makes the program iterate to find a right line to seperate two classes based on Perceptron algorithm. <br />
![Perceptron_lseperable](/images/Perceptron_lseperable.gif)

In a typical non-linearly seperable case (XOR), it goes as below when the number of iterations hits the number you set. <br />
![Perceptron_nlseperable](/images/Perceptron_nlseperable.gif) <br />
<br />
<br />

### Linear Gaussian Naive Bayes Classifier
When each of those decomposed one-dimension feature distributions, $P(x_i | y)$ follows Gaussian distribution and their variances are assumed to be the same across all labels (in this case when y=1 and y=-1), we can derive the closed form of a line that represents 50:50 case of falling into the two labels. In other words, Gaussian Naive Bayes classifier's decision boundary is linear in this case. <br />

![linear_GNB](/images/linear_GNB.gif) <br />
<br />
<br />

### Logistic Regression Classifier
Logistic regression is the discriminative counterpart to linear Gaussian Naive Bayes, which means it directly estimates w and b in the equation, $P(y=1 | x)={1\over{1+ w^{T}*x + b}}$. Linear Gaussian Naive Bayes also estimates w and b but with Bayesian form, by decomposing $P(X | y)$ into each feature distribution, $P(x_{i} | y)$ and assumming independent Gaussian distribution for each feature distribution. Surprisingly when GNB is linear, the final form of these two ML algorithms are the same as $P(y=1 | x)={1\over{1+ w^{T}*x + b}}$ (for the proof, refer to [this](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf))

When it comes to logistic regression, it doesn't have to assume any form of feature distribution given label so it is more flexible than linear Gaussian Naive Bayes but requires more data. And also since there is no closed form of the derivative equation of w, minimum-finding-algorithms such as Gradient Descent and Newton's method are used to find the minimum of the logistic regression loss function. 


## Setup

### git clone
git clone to have this repository on your local machine as follows.
```ruby
git clone git@github.com:YANJINI/Linear-Classifiers-with-Visualization-on-A-Plane.git
```

### path control
To import modules written in this repository on your local macine, you need control path to this clone, which could be done as below. (Mac OS)
```ruby
import sys
sys.path.extend(['/Users/jinijani/PycharmProjects/Practice/git_project/Single-Perceptron-and-Gaussian-Naive-Bayes-Classifier-with-Visualization-on-A-Plane'])
```

### import 
Import these two classifiers in another py project as below.
```ruby
from Perceptron import twoD_coordinates_Perceptron
from linear_GNB import twoD_coordinates_lGNB
from logistic_r import twoD_coordinates_logistic_r
```

### Others
Check examples.py to see how to use the programes
