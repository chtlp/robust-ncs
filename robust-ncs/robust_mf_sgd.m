function [U, V, models] = robust_mf_sgd(M, W, dim, round, options, U0, V0)

%learn_rate = options.learn_rate;
%humber_lambda = options.huber_lambda;
%L2Reg = options.L2Reg;
%L1Reg = options.L1Reg;

display_freq = options.display_freq;
save_model_freq = options.save_model_freq;
symm_factorization = options.symm_factorization;
BoldDrive = options.BoldDrive;

neighbor_list = create_neighbor_list(W);
[training_samples, ~] = size(neighbor_list);

[n, ~] = size(M);

if nargin == 5
    U = randn(n, dim) * 0.1;
    V = randn(n, dim) * 0.1;
else
    U = U0;
    V = V0;
end

models(round) = struct;
for i = 1:round
    neighbor_list = neighbor_list(randperm(training_samples), :);
	
    [loss, complexity, U1, V1] = iterate_huber_sgd(M, neighbor_list, U, V, options);
    if mod(i, save_model_freq) == 0
        models(i).U = U;
        models(i).V = V;
    end
    
	if symm_factorization
		P = U * V' + V * U';
    else
		P = U * V';
	end
	
    E = abs(P - M);
    E2 = E.^2;
    mae = mean(E(M > 0));
    rmse = mean(E2(M > 0));
    train_mae = mean(E(W));
    train_rmse = mean(E2(W));

    models(i).mean_loss = loss / training_samples;
    models(i).complexity = complexity;
    models(i).loss = loss;
    
    models(i).mae = mae;
    models(i).train_mae = train_mae;
    models(i).rmse = rmse;
    models(i).train_rmse = train_rmse;
        
    if BoldDrive
        if i > 1 && models(i-1).loss+models(i-1).complexity <= loss + complexity
            options.learn_rate = options.learn_rate / 2;
        else
            options.learn_rate = options.learn_rate * 1.05;
            U = U1; V = V1;
        end
        
        if options.learn_rate < 1e-6
            break
        end
    else
        U = U1; V = V1;
    end
    
    
    if mod(i, display_freq) == 0
        fprintf('round %d, cost = %.3e, loss = %.3e, train-huber = %.3f, tmae = %.3f, mae = %.3f\n', ...
            i, loss+complexity, loss, loss / training_samples, train_mae, mae);
        if BoldDrive
            fprintf('learn-rate = %.3e \n', options.learn_rate);
        end
    end
end
end

function list = create_neighbor_list(W)
[m n] = size(W);
list = zeros(sum(sum(W > 0)), 2);
k = 0;

for i = 1:m
	for j = 1:n
		if W(i,j) > 0
			k = k + 1;
			list(k,:) = [i, j];
		end
	end
end		
end