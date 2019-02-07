F = imread('0.jpg');
F = double(F);
F_unrolled = reshape(F,1,400);
mu = (sum(F_unrolled))/(length(F_unrolled));
SD = std(F_unrolled);
F_unrolled = (F_unrolled - mu)/SD;
m = size(F_unrolled, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(F_unrolled, 1), 1);

h1 = sigmoid([ones(m, 1) F_unrolled] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2')
[dummy, p] = max(h2, [], 2)
