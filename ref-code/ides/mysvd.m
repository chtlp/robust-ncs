function [U, V] = mysvd(D, d)
%result should be D = U * V
%  d is the lower dimension
%  D is NxN matrix

    N = length(D);
    
    [U W V] = svd(D);
    % disp(W);
    sq = sqrt(W);
    U = U*sq;
    U = U(:, 1:d);
    V = sq*V';
    V = V(1:d, :);
    return
