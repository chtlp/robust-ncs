function [U, V, models] = robust_mf_alternative(M, W, dim, round, options)

assert(all(all(W' == W)));
echo_freq = options.echo_freq;
max_iter = options.max_iter;
lambda = options.huber_lambda;
save_freq = options.save_freq;

[m, n] = size(M);
assert(m == n);


deviation = 1;
U = randn(m, dim) * deviation;
V = randn(m, dim) * deviation;

models(round) = struct();

for iter = 1:round
%     flag = zeros(m,1);

    order = randperm(2*m);
    
    collector.k = 0;
    collector.length_ratio = zeros(10000, 1);
    
    for k = 1:length(order)
        if order(k) <= m
            i = order(k);
            b = M(i, W(i,:))';
            A = U(W(i,:),:);            
            [V(i,:), fval, num_iter, r] = updated_vector3(A, b, V(i,:), lambda, max_iter);
        else
            i = order(k) - m;
            b = M(i, W(i,:))';
            A = V(W(i,:),:);            
            [U(i,:), fval, num_iter, r] = updated_vector3(A, b, U(i,:), lambda, max_iter);
        end
        
        collector.k = collector.k + 1;
        collector.length_ratio(collector.k) = r;
        if mod(k, echo_freq) == 0
            fprintf('round = %d, k = %d, loss = %.2f, num_iter = %d, time = %s\n', iter, k, fval, num_iter, datestr(now));
        end
    end

	E = abs(U * V' - M);
	train_mae = mean(E(W));
	mae = mean(E(M>0));
	mean_loss = mean(my_huber(E(W), lambda));
	
	models(iter).mean_loss = mean_loss;
	models(iter).mae = mae;
	models(iter).train_mae = train_mae;
	
	if mod(iter, save_freq) == 0
		models(iter).U = U;
		models(iter).V = V;
	end
	
    fprintf('iter = %d, mean_loss = %.3f, train-mae = %.3f, mae = %.3f\n', iter, mean_loss, train_mae, mae);
    L = collector.length_ratio;
    assert(all(L == 0 | L > 1e-2));
    fprintf('mean-length-ratio: %e\n', mean(L(L > 0)));
    
end
end

function Y = my_huber(X, lambda)
X = abs(X);
Y = (X < lambda) .* (X.^2 / (2*lambda)) + (X >= lambda) .* (X - lambda / 2);
end