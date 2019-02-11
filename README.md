# Machine Learning - Neural Networks Learning

#### <em>Aim: Build and train a neural network (using the backpropagation algorithm) for the purpose of hand-written digit recognition.</em><br>

Training set used was a dataset of 5000 20x20 pixel grayscale images of hand-written digits (source: MNIST).<br>

##### [ex3data1.mat](https://github.com/fvarnals/Neural-Networks-Learning/blob/master/ex3data1.mat) contains:<br>
<code>X</code> - 5000x400 Matrix , where each row is a 400 dimensional vector representing a single training example for a handwritten digit image, created by 'unrolling' the 20x20 grid of pixels for each digit image.<br>

<code>y</code> - 5000 dimensional vector that contains labels for the training set. Because there is no zero index in MATLAB, the digit zero has been mapped to the value '10'. Therefore, a “0” digit is labeled as “10”, while the digits “1” to “9” are labeled as “1” to “9” in their natural order.<br>

The neural network that we use is shown in the figure below, and has the following architecture: an <strong>input layer</strong>, one <strong>hidden layer</strong>, and an <strong>output layer</strong>, with the <code>sigmoid</code> activation function (<code>g</code>) applied to the outputs of each layer.<br>
<img src="https://github.com/fvarnals/Neural-Networks-Learning/blob/master/network_architecture.png" width=300 ><br>

The input values are the pixel values of the digit images, and therefore the input layer has 400 units, plus an extra 'bias' unit which always inputs +1.<br>

The aim is to calculate the optimum Theta values (Theta 1 and Theta 2 matrices) for all of the network connections, such that the network can accurately return the correct label of each handwritten digit image.<br>

To achieve this, we first need to calculate the cost function of the neural network, so that we can run gradient descent and find the vlues of Theta which minimise the cost, thereby training the network to accurately categorise the digit images.

1) <strong>[nnCostFunction.m](https://github.com/fvarnals/Neural-Networks-Learning/blob/master/nnCostFunction.m) - Feedforward using initial Theta values, so that we can calculate the cost function using backpropagation.

<em>sigmoidGradient.m</em></strong> - Compute the gradient of the sigmoid function<br>
<em>randInitializeWeights.m</em></strong> - Randomly initialize weights<br>
<em>nnCostFunction.m</em></strong> - Neural network cost function<br>
