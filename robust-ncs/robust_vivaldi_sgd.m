function [C, mae_seq] = robust_vivaldi_sgd(M, W, dim, lambda, max_iter, opt)

sample_list = make_sample_list(W);
num_samples = size(sample_list, 1);
N = size(W, 1);

learn_rate = N / num_samples;
iter = 0;
last_loss = inf;

C = rand(N, dim+1);

if nargout > 1
    mae_seq = nan(max_iter, 1);
end

while learn_rate > 1e-7 && iter < max_iter
    iter = iter + 1;
    sample_list = sample_list(randperm(num_samples), :);
    
        [C1, loss] = robust_vivaldi_height_sgd_iterate(M, sample_list, C, learn_rate, lambda);
    
    if loss < last_loss
        last_loss = loss;
        C = C1;
        learn_rate = learn_rate * 1.05;
    else
        learn_rate = learn_rate * 0.5;
    end
    
    if mod(iter, opt.display_round) == 0
        fprintf('iter = %d, mean-loss = %.6f, learn_rate = %.6f\n', iter, loss / num_samples, learn_rate);
        if (opt.calc_error)
            P = dist_matrix(C, true);
            E = abs(P-M);
            mae_train = mean(E(W));
            mae_test = mean(E(M >= 0));
            fprintf('\ttrain-mae = %.3f, mae = %.3f\n', mae_train, mae_test);
            if nargout > 1
                mae_seq(iter) = mae_test;
            end
        end
    end
end

end

function list = make_sample_list(W)

N = size(W, 1);
K = sum(sum(W > 0));
list = zeros(K, 2);
k = 0;
for i = find(W > 0)'
    k = k + 1;
    x = mod(i, N);
    if x == 0
        x = N;
    end
    y = (i-x) / N + 1;
    list(k,:) = [x y];
end

end

