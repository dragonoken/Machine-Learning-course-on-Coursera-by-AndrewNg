function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Using an algebra fact : (a - b)^2 = a^2 - 2ab + b^2
% minimize : - 2ab + b^2 (in this case, a^2 does not matter)

centroids_squared = zeros(size(centroids, 1), 1);
for cent_i = 1:size(centroids, 1)
  centroids_squared(cent_i) = centroids(cent_i, :) * centroids(cent_i, :)';
endfor

[~, idx] = min(centroids_squared' - (2 * X * centroids'), [], 2);

% =============================================================

end

