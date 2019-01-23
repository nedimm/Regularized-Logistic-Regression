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