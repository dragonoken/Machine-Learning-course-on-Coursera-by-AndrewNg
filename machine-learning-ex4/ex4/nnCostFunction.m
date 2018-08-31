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
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
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


% Transform y into a matrix of one-hot vectors
% Examples are in columns, rows corresponds to each diffrent label values
oneHot_y = zeros(num_labels, size(y, 1));
for example = 1:size(y, 1)

  oneHot_y(y(example), example) = 1;

endfor


% Forward Propagation (Feedforward)
inputLayer = [ones(m, 1) X]'; % Each column is an example with every first value being the bias unit

hiddenLayer = [ones(1, m); sigmoid(Theta1 * inputLayer)];

outputLayer = sigmoid(Theta2 * hiddenLayer);


% Cost(J) Calculation (With Regularization)
% To sum all individual cost for every example and every label, unroll the matrices and do the multiplications
% An alternative is using element-wise multiplications and then sum the values using sum function (probably twice)
J = (-oneHot_y(:)' * log(outputLayer)(:) - (1 - oneHot_y)(:)' * log(1 - outputLayer)(:)) / m ...
      + (Theta1(:,2:end)(:)' * Theta1(:, 2:end)(:) + Theta2(:,2:end)(:)' * Theta2(:, 2:end)(:)) * (lambda / (2 * m));


% Gradient Calculation (Backpropagation)
% Initialize Deltas (Accumulative Gradient Values)
Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
% For each example
for example = 1:m

  % Feedforward
  a_1 = [1, X(example,:)]';

  z_2 = Theta1 * a_1;
  
  a_2 = [1; sigmoid(z_2)];

  z_3 = Theta2 * a_2;

  a_3 = sigmoid(z_3);

  % Backpropagate
  delta_3 = a_3 - oneHot_y(:,example);

  delta_2 = (Theta2(:,2:end)' * delta_3) .* sigmoidGradient(z_2);

  Delta_2 = Delta_2 + (delta_3 * a_2');

  Delta_1 = Delta_1 + (delta_2 * a_1');

endfor
% Devide the accumulated gradients by the number of examples
D_2 = Delta_2 / m;

D_1 = Delta_1 / m;

Theta1_grad = D_1;

Theta2_grad = D_2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
