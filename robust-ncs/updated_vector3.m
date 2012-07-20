function [X, loss, num_iter, r] = updated_vector3(A, b, X0, lambda, max_iter)

delta = 1;
X = X0';
dim = size(X, 1);
r = 0;
for i = 1:max_iter
    H = A' * huber_h(A*X-b, lambda) * A;
    G = A' * huber_g(A*X-b, lambda);
    
    while true
        dX = -(H+delta * eye(dim)) \ G;
        if huber(A*(X+dX)-b, lambda) < huber(A*X-b, lambda)
            delta = delta * 0.5;
            % only record the last ratio
            r = norm(dX, 2) / norm(X, 2);
            if r > 1e-2
                X = X+dX;
            else
                r = 0;
            end
            break
        else
            delta = delta * 2;
            if delta > 100
                break
            end
        end
    end
end

loss = huber(A*X-b, lambda);
num_iter = i;
X = X';
end

function y = huber(X, lambda)
y = (abs(X) < lambda) .* (X.^2 /2/lambda) + (abs(X) >= lambda) .* (abs(X) - lambda/2);
y = sum(y);
end

function g = huber_g(X, lambda)
g = (abs(X) < lambda) .* X/lambda + (abs(X) >= lambda) .* sign(X);
end

function h = huber_h(X, lambda)
h = (abs(X) < lambda) / lambda;
h = diag(h);
end