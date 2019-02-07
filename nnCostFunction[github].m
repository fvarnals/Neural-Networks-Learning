function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

input_values = [ones(size(X,1),1) X];
hidden_layer_values = input_values*Theta1';
hidden_layer_outputs = sigmoid(hidden_layer_values);
hidden_layer_outputs = [ones(size(hidden_layer_outputs,1),1) hidden_layer_outputs];
output_layer_values = hidden_layer_outputs * Theta2';
output_layer_outputs = sigmoid(output_layer_values);

y_matrix = eye(num_labels)(y,:);

error_matrix = ((-y_matrix'*log(output_layer_outputs))-((1-y_matrix)'*log(1-output_layer_outputs)));
% the above basically outputs a 10x10 matrix (if output and y_matrix are 5000x10)
% so what we get is a 10x10 matrix which is the sum of errors for every row compared
%to eachother (i.e output vs errors)
%but we don't want every row compared, we only want row x compared to row x
%i.e we don't want the sum of errors between row 3 and row 72 etc etc etc
% SO we use the identity matrix below, to give only the diagonal values in the y_matrix
% i.e the sums of errors between row 1 of output and row 1 of y, row 17 of output
% and row 17 of y. Therefore we ONLY want the diagonal values of the error_matrix
% we then use a double sum to get the sum of all the

error = eye(size(error_matrix)).*error_matrix;

J = (1/m) * sum(sum((error)));

Theta1 = Theta1(:,2:end);
Theta2 = Theta2(:,2:end);

Theta1_reg = sum(sum(Theta1.^2));

Theta2_reg = sum(sum(Theta2.^2));

reg_term = Theta1_reg + Theta2_reg;

J = J + ((lambda/(2*m)) * reg_term);


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
I = eye(num_labels);
Y = zeros(m, num_labels);
for i = 1:m
  Y(i, :) = I(y(i), :);
end

A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
Z3 = A2 * Theta2';
H = A3 = sigmoid(Z3);

% step 1 Feedforward and cost function
J = (1 / m) * sum(sum((-Y) .* log(H) - (1 - Y) .* log(1 - H), 2));

% step 2 Regularized cost function
penalty = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2, 2)) + sum(sum(Theta2(:, 2:end) .^ 2, 2)));
J = J + penalty;

% step 3 Neural Network Gradient (Backpropagation)
Sigma3 = A3 - Y;
Sigma2 = (Sigma3 * Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);

Delta_1 = Sigma2' * A1;
Delta_2 = Sigma3' * A2;

% step 4 Regularized Gradient
Theta1_grad = Delta_1 ./ m + (lambda / m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = Delta_2 ./ m + (lambda / m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];





%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
