d3_matrix = zeros(5000,10);
d2_matrix = zeros(5000,25);
for t = 1:m
  data = X(t, :); % 1x400
  input_values = [1, data]; % 1x401
  Theta_1 = randInitializeWeights(input_layer_size, hidden_layer_size); %25x401
  hidden_layer_values = input_values*Theta_1'; %1x401 * 401x25 = 1x25
  hidden_layer_outputs = sigmoid(hidden_layer_values); % 1x25
  hidden_layer_outputs = [1, hidden_layer_outputs]; % 1x26

  Theta_2 = randInitializeWeights(hidden_layer_size, num_labels); % 10x26

  output_layer_values = hidden_layer_outputs * Theta_2'; % 1x26 * 26x10 = 1x10
  output_layer_outputs = sigmoid(output_layer_values); % 1x10

  num_label = y(t,1); % get number label for row in question from y 1x1
  y_value_vector = zeros(1, num_labels); % setting up logical array template 1x10
  y_value_vector(1,num_label) = 1; % place value '1' in column relating to num_label 1x10

  d3 = output_layer_outputs - y_value_vector; %1x10
  d3_matrix(t, :) = d3;  %5000x10
  Theta_2 = Theta_2(:,2:end);
  d2 = (d3 * Theta_2) .* sigmoidGradient(hidden_layer_values); %??? correct? - think so, gives 1x25
  d2_matrix(t,:) = d2;
end
