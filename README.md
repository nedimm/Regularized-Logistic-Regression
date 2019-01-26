# Regularized Logistic Regression

In this project we will implement regularized logistic regression model to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure
it is functioning correctly. 
The project is an exercise from the ["Machine Learning"](https://www.coursera.org/learn/machine-learning/) course from Andrew Ng.

The task is described as follows: 
Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. From these two tests, you would like to determine whether the microchips should be accepted or
rejected. To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.
You have historical data from previous test result that you can use as a training set stored in the `ex2data2.txt` file.

The implementation was done using [GNU Octave](https://www.gnu.org/software/octave/). The start point is the `ex2_reg.m` script and other functions are implemented in separate `*.m` files.

## Visualizing the data
Before starting to implement any learning algorithm, it is always good to visualize the data if possible. The figure below displays the historical data where the axes are the two test scores, and the positive (y = 1, accepted) and negative (y = 0, rejected) examples are shown with different markers.

![viz](https://i.imgur.com/7X1j3az.png)
*Figure 1: Training Data*

Figure 1 shows that our dataset cannot be separated into positive and negative examples by a straight-line through the plot. Therefore, a straightforward application of logistic regression will not perform well on this dataset since logistic regression will only be able to find a linear decision boundary.

## Feature mapping
One way to fit the data better is to create more features from each data point. In the provided function `mapFeature.m`, we will map the features into all polynomial terms of x1 and x2 up to the sixth power.

![fm](https://i.imgur.com/8ubWgPP.png)

As a result of this mapping, our vector of two features (the scores on two QA tests) has been transformed into a 28-dimensional vector. A logistic regression classifier trained on this higher-dimension feature vector will have
a more complex decision boundary and will appear nonlinear when drawn in our 2-dimensional plot.
While the feature mapping allows us to build a more expressive classifier, it also more susceptible to overfitting. To address that, we will implement regularized logistic regression to fit the data and also see how regularization can help combat the overfitting problem.

## Cost function and gradient
The regularized cost function in logistic regression is:

![rc](https://i.imgur.com/hZwQxpy.png)

Note that we should not regularize the parameter θ0. In Octave/MATLAB, recall that indexing starts from 1, hence, we should not be regularizing the theta(1) parameter (which corresponds to θ0) in the code. The gradient of the cost function is a vector where the jth element is defined as follows:

![gradient0](https://i.imgur.com/FO3Jep0.png)
![gradient1](https://i.imgur.com/f5C8p6z.png)

The implementation of the cost function looks like:

```matlab
function [J, grad] = costFunctionReg(theta, X, y, lambda)
  %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
  %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters. 

  m = length(y); % number of training examples
  grad = zeros(size(theta));
  J = ( (1 / m) * sum(-y'*log(sigmoid(X*theta)) - (1-y)'*log( 1 - sigmoid(X*theta))) ) + (lambda/(2*m))*sum(theta(2:length(theta)).*theta(2:length(theta))) ;
  grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) );
  grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';
end
```
After we implement the cost function, we call it with initial value of θ (initialized to all zeros). We should see that the cost is about 0.693. We will use now the fminunc function to learn the optimal parameters θ.

![output](https://i.imgur.com/QQKiPWO.png)

In order to visualize the model learned by this classifier, we plot the (non-linear) decision boundary that separates the positive and negative examples. In `plotDecisionBoundary.m`, we plot the non-linear decision boundary by computing the classifier’s predictions on an evenly spaced grid and then drew a contour plot of where the predictions change from y = 0 to y = 1. After learning the parameters θ, the next step in `ex_reg.m` will plot a decision boundary similar to Figure 2.

![viz](https://i.imgur.com/TVxhFyJ.png)
*Figure 2: Training data with decision boundary*

# Preventing Overfitting
 Now we can try out different regularization parameters for the dataset to understand how regularization prevents overfitting.
Notice the changes in the decision boundary as you vary λ. With a small λ, you should find that the classifier gets almost every training example correct, but draws a very complicated boundary, thus overfitting the data
(Figure 2). This is not a good decision boundary: for example, it predicts that a point at x = (−0:25;1:5) is accepted (y = 1), which seems to be an incorrect decision given the training set.
With a larger λ, you should see a plot that shows an simpler decision boundary which still separates the positives and negatives fairly well (Figure 3). However, if λ is set to too high a value, you will not get a good fit and the decision
boundary will not follow the data so well, thus underfitting the data (Figure 4).

![overfitting](https://i.imgur.com/4eLbdoB.png)
*Figure 3: Training data with decision boundary (λ = 1)*


![underfitting](https://i.imgur.com/uaPLm1Z.png)
*Figure 4: Too much regularization (Underfitting) (λ = 100)*