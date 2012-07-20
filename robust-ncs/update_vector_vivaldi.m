function [X, loss, num_iter] = update_vector_vivaldi(A, b, X0, lambda)

options = optimset('GradObj','on', 'Display', 'off', 'MaxIter', 2); % indicate gradient is provided 
[X, fval, exitflag, output] = fminunc(@(x) my_huber(x, A, b, lambda), X0,options);

[f, g] = my_huber(X0, A, b, lambda);
loss0 = sum(f);

[f, g] = my_huber(X, A, b, lambda);
loss = sum(f);

if loss >= loss0
    fprintf('loss = %.3f loss0 = %.3f\n', loss, loss0);
end

num_iter = output.iterations;

end

function [f, g] = my_huber(X, A, b, lambda)
[m n] = size(A);
delta = repmat(X, m, 1) - A;
dist = sqrt(sum(delta.^2,2));
r = dist - b;
f = sum( r.^2/(2*lambda) .* (abs(r) <= lambda) + (abs(r) - lambda/2) .* (abs(r) > lambda) );
huber_g = r/lambda .* (abs(r) <= lambda) + sign(r) .* (abs(r) > lambda);
g = delta' * (huber_g ./ dist);
end