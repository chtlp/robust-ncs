function P = runPhoenix( M, neighbors, dim, num_round )
addpath ../ref-code/phoenix

round_length = 1;
session_length = 10^7;
[out_host, in_host] = simple_phoenix_churn(M, dim, length(M), neighbors, 10, num_round, round_length, session_length);
rmpath ../ref-code/phoenix

P = out_host * in_host;
end