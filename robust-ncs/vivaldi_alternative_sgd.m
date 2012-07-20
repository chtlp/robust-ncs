dim = 2;
round = 100;
lambda = 7;
opt.display_error_freq = 1;
opt.update_debug = false;
opt.update_max_iter = 20;
opt.update_max_updates = 3;

[C, alternative_mae_seq] = robust_vivaldi_alternative(M, W | W', dim, round, lambda, opt);

%%
load('alternative_sgd_seq.mat');

%%
round = 100;
opt.display_round = 1;
opt.calc_error = true;
opt.use_height = true;
[C2, sgd_mae_seq] = robust_vivaldi_sgd(M, W, dim, lambda, round, opt);


%save('alternative_sgd_seq.mat', 'alternative_mae_seq', 'sgd_mae_seq');

%%
plot(1:100, sgd_mae_seq, 'r--', 1:100, alternative_mae_seq, 'b-');
xlabel('round'); ylabel('MAE (ms)');
axis([-inf inf 0 100]);
legend('Stochastic Gradient Descent', 'Alternative Damped-Newton Method');
savefig('alternative_sgd_convergence','pdf');

