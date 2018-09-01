function [error_train, error_val] = ...
    randLearningCurve(X, y, Xval, yval, lambda)
%RANDLEARNINGCURVE Generates the train and cross validation set errors needed using randomly selected samples
%to plot a learning curve
%   [error_train, error_val] = ...
%       RANDLEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% Number of validation examples
m_val = size(Xval, 1);

% Number of iterations per each number of examples
iterations = 50;

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);


for subset_size = 1:m

  for iter = 1:iterations

    randIndex = randperm(m, subset_size);

    randIndex_val = randperm(m_val, min([subset_size, m_val]));

    XrandSub = X(randIndex, :);

    yrandSub = y(randIndex);

    XrandVal = Xval(randIndex_val, :);

    yrandVal = yval(randIndex_val);

    [theta] = trainLinearReg(XrandSub, yrandSub, lambda);

    error_train(subset_size) = error_train(subset_size) + (linearRegCostFunction(XrandSub, yrandSub, theta, 0) / iterations);

    error_val(subset_size) = error_val(subset_size) + (linearRegCostFunction(XrandVal, yrandVal, theta, 0) / iterations);

  endfor

endfor

% =========================================================================

end
