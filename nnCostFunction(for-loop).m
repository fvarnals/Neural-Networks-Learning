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
Delta1 = zeros(hidden_layer_size,1);
Delta2 = zeros(num_labels, 1);
Theta_1_bp = randInitializeWeights(input_layer_size, hidden_layer_size); %25x401
Theta_2_bp = randInitializeWeights(hidden_layer_size, num_labels); % 10x26

for t = 1:m
  data = X(t, :); % 1x400
  a1 = [1, data]; % 1x401
  z2 = a1*Theta_1_bp'; %1x401 * 401x25 = 1x25
  a2 = sigmoid(z2); % 1x25
  a2_wbias = [1, a2]; % 1x26

  z3 = a2_wbias * Theta_2_bp'; % 1x26 * 26x10 = 1x10
  a3 = sigmoid(z3); % 1x10

  num_label = y(t,1); % get number label for row in question from y 1x1
  y_value_vector = zeros(1, num_labels); % setting up logical array template 1x10
  y_value_vector(1,num_label) = 1; % place value '1' in column relating to num_label 1x10

  d3 = a3 - y_value_vector; %1x10
  Theta_2_bp_trunc = Theta_2_bp(:,2:end);
  d2 = (d3 * Theta_2_bp_trunc) .* sigmoidGradient(z2); %1x10 * 10x25, gives 1x25

  Delta1 = Delta1 + (d2' * a1); % 1x25' * 1x401
  Delta2 = Delta2 + (d3' * a2_wbias); %1x10' * 1x26
end

%Implement regularization with the cost function and gradients.
Theta1_grad = (1/m) * (Delta1 + (lambda*Theta_1_bp));
Theta2_grad = (1/m) * (Delta2 + (lambda*Theta_2_bp));


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
