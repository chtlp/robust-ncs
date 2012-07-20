function [X1, loss] = update_vector_vivaldi_height(A, b, X, huber_lambda, opt)

debug_on = opt.update_debug;
max_iter = opt.update_max_iter;
max_updates = opt.update_max_updates;


dim = size(A,2) - 1;
AH = A(:,end);
A = A(:,1:end-1);
XH = X(end);
X = X(1:end-1);
assert(all(AH >= 0) && XH >= 0);

[loss0, G, H] = compute_loss(A, AH, X, XH, b, huber_lambda);

lambda = 0.1;
last_loss = loss0;

if debug_on
    fprintf('loss(X0) = %.3f\n', last_loss);
end

iter = 0;
updates = 0;
while lambda < 10^6 && iter < max_iter && updates < max_updates
    iter = iter + 1;
    
    X1 = [X XH] - G / (H+lambda*diag(ones(dim+1,1))) ;
    X1H = max(X1(end), 0);
    X1 = X1(1:end-1);
        
    loss = compute_loss(A, AH, X1, X1H, b, huber_lambda);
    
    if debug_on
        fprintf('\titer = %d, lambda = %.3f, loss = %.3f\n', iter, lambda, loss);
    end
    
    if loss < last_loss
        X = X1; XH = X1H;
        [last_loss, G, H] = compute_loss(A, AH, X, XH, b, huber_lambda);
        lambda = lambda / 1.05;
        updates = updates + 1;
    else
        lambda = lambda * 2;
    end
end

loss1 = compute_loss(A, AH, X, XH, b, huber_lambda);
assert(loss1 <= loss0);

X1 = [X XH];
loss = last_loss;
end

function [f, g, h] = my_huber(X, lambda)
absX = abs(X);
f = (absX < lambda) .* (absX.^2 / 2 / lambda) + (absX >= lambda) .* (absX - lambda / 2);

if nargout > 1
    g = (absX < lambda) .* (X / lambda) + (absX >= lambda) .* sign(X);
    h = (absX < lambda) / lambda;
end
end

function [F, G, H] = compute_loss(A, Ah, X, Xh, b, lambda)
eps = 1e-3;

[N,dim] = size(A);
assert(all(Ah >= 0) && Xh >= 0);
dist = sqrt(sum((A - repmat(X,N,1)).^2,2));
P = dist + Ah + Xh;
[huber_f, huber_g, huber_h] = my_huber(P-b, lambda);

F = sum(huber_f);

if nargout > 1
    delta_g = (repmat(X,N,1) - A) ./ repmat(dist + eps, 1, dim);
    G = sum(repmat(huber_g, 1, dim) .* delta_g);
    H = delta_g' * diag(huber_h) * delta_g + diag(ones(dim,1)) * sum(huber_g ./ dist);

    G = [G sum(huber_g)];
    dxdh = sum(repmat(huber_h, 1, dim) .* delta_g);
    H = [H dxdh'; dxdh sum(huber_h)];
end
end