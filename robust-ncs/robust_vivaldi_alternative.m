function [C, mae_seq] = robust_vivaldi_alternative( M, W, dim, round, lambda, opt, C )

m = size(M, 1);

if nargin < 7
    C = rand(m, dim+1);
end

if nargout > 1
    mae_seq = nan(round,1);
end

for iter = 1:round

    order = randperm(m);
    loss = zeros(m, 1);
    for k = 1:length(order)
        i = order(k);
        neighbors = W(i,:);
        assert(neighbors(i) == false);
        assert(all(M(i,neighbors) > 0));
        b = M(i, neighbors)';
        A = C(neighbors,:);            
        [C(i,:), fval] = update_vector_vivaldi_height(A, b, C(i,:), lambda, opt);
        loss(k) = fval;
    end
    fprintf('iter = %d, mae-loss = %.3f\n', iter, sum(loss) / sum(sum(W)));
    
    if mod(iter, opt.display_error_freq) == 0
        P = dist_matrix(C, true);
        E = abs(P - M);
        EH = my_huber(E, lambda);
        fprintf('\ttrain-mae = %.3f, mae = %.3f\n', mean(E(W > 0)), mean(E(M >= 0)));
        fprintf('\ttrain-huber = %.3f, huber = %.3f\n', mean(EH(W > 0)), mean(EH(M >= 0)));
    
        if nargout > 1
            mae_seq(iter) = mean(E(M >= 0));
        end
    end
end
end

function Y = my_huber(X, lambda)
X = abs(X);
Y = (X < lambda) .* (X .^ 2 / (2*lambda)) + (X >= lambda) .* (X - lambda/2);
end