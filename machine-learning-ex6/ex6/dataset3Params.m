function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cs = [0.1, 0.3, 1, 3, 10, 30];
n_Cs = numel(Cs);

sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3];
n_sigmas = numel(sigmas);

val_errors = zeros(n_Cs * n_sigmas, 1);

fprintf('\n');
for C_i = 1:n_Cs
  for sigma_i = 1:n_sigmas
    fprintf('%d of %d parameter combinations (C : %f, sigma : %f)', ...
               (n_sigmas * (C_i - 1) + sigma_i), (n_Cs * n_sigmas), Cs(C_i), sigmas(sigma_i));
    model= svmTrain(X, y, Cs(C_i), @(x1, x2) gaussianKernel(x1, x2, sigmas(sigma_i)));
    predictions = svmPredict(model, Xval);
    val_errors(n_sigmas * (C_i - 1) + sigma_i) = mean(double(predictions ~= yval));
  endfor
endfor

best_index = find(val_errors == min(val_errors))(1);

C = Cs(floor(best_index / n_sigmas) + 1);
sigma = sigmas(mod(best_index, n_sigmas));

fprintf('\nBest Parameters : (C : %f, sigma : %f)\n', C, sigma);
fprintf('Best Error : %f\n', val_errors(best_index));

% =========================================================================

end
