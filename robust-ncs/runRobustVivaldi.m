function P = runRobustVivaldi(M, W, dim, round, lambda)

% opt.display_round = 2000;
% opt.calc_error = false;
% opt.use_height = true;
% C = robust_vivaldi_sgd(M, W, dim, lambda, round, opt);
% 
% P = dist_matrix(C, true);

%dim = 2;
%round = 40;
%lambda = 7;
opt.display_error_freq = 5;
opt.update_debug = false;
opt.update_max_iter = 20;
opt.update_max_updates = 3;

[C, mae_seq] = robust_vivaldi_alternative(M, W | W', dim, round, lambda, opt);

P = dist_matrix(C, true);

end