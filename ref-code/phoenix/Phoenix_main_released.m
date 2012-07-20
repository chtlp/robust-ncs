function Phoenix_main_released()

% raw distance matrix %
clear
load('recon.mat');
load king_matrix.txt

%seq=randperm(3997);



DATA=PL;
%DATA=king_matrix;

default_dimension = 8;

max_round = 5;



evaluate_churn = 1;





if (evaluate_churn == 1)

    round_length_seq = [100 200 300 400 500 600 700 800 900 1000]; % seconds
    avg_session_length = [600 1200 1800]; % seconds
    avg_seq_npre_matrix = [];
    for diff_session_length = 1:length(avg_session_length)
        fprintf('Avg Session Length = %ds\n', avg_session_length(diff_session_length));
        npre_matrix = [];
        for round=1:max_round
            npre_seq = [];
            for i=1:length(round_length_seq)
                [out_host, in_host, stabilized_npre] = phoenix_churn(DATA, default_dimension, length(DATA), 32, 5, round_length_seq(i), avg_session_length(diff_session_length)); 
                npre_seq= [npre_seq stabilized_npre];
            end
            npre_seq
            npre_matrix = [npre_matrix; npre_seq];
        end        
        avg_seq_npre_matrix = [avg_seq_npre_matrix; mean(npre_matrix)];
    end
    
    figure;
    avg_seq_npre_matrix
    h1=plot(round_length_seq, avg_seq_npre_matrix(1, :), '-+');set(h1, 'LineWidth', 2);hold on;
    h2=plot(round_length_seq, avg_seq_npre_matrix(2, :), '--o');set(h2, 'LineWidth', 2);hold on;
    h3=plot(round_length_seq, avg_seq_npre_matrix(3, :), ':x');set(h3, 'LineWidth', 2);hold on;
    if (length(DATA) == 169)
        axis([100 1000 0 3]);
    else
        axis([100 1000 0 1.6]);
    end
    xlabel('Update Interval (sec)', 'FontSize', 16);ylabel('90th Relative Error', 'FontSize', 16);
    h5 = legend('Avg Time = 10min', 'Avg Time = 20min', 'Avg Time = 30min', 2);set(h5, 'FontSize', 16);
    RE_filename = 'Phoenix_Churn_';
    tmp_size = length(DATA);
    RE_filename = strcat(RE_filename, num2str(tmp_size));
    saveas(gcf, RE_filename, 'eps');
end
