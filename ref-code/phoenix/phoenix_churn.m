function [out_host, in_host, stabilized_npre] = phoenix_churn(D, dim, N, K, C, round_length, mean_on_length)

% N: Number of nodes
% dim: dimension of the NC
% K: Number of Neighbors
% C: 10
% round_length: NC update interval
% mean_on_length: average session length

D_change = D;

% Parameters
C = 10;
%new_host_scale = 20;

% Per Round -> Per Second ...


round_bound = 50;

% 30*64 = 
closest_num =0;
result_matrix_seq = [];


avg_up_length = 10;
avg_down_length = 10;

% ignore it, just one node?
if (N == 1)
%    [out_host, in_host] = NMF(D, dim);
    out_host = zeros(1, dim);
    in_host = zeros(dim, 1);
    result_matrix_seq = [];
    if (converge_on == 1)
        predicted_matrix = out_host*in_host;
        for round=1:round_bound
            result_matrix_seq = [result_matrix_seq; D(1, 1)];
        end
    end
    return
end

% ignore it, less than K nodes?
if (N < K)
    % NMF Directly %
    length(D);
    [out_host, in_host] = NMF(D, dim);
    result_matrix_seq = [];
    if (converge_on == 1)
        predicted_matrix = out_host*in_host;
        for round=1:round_bound
            result_matrix_seq = [result_matrix_seq; predicted_matrix];
        end
    end
    
   return;
end

% for N >= K, K is the number of neighbors
neighbor = zeros(N,K);
error_in = zeros(1,N);
error_out = zeros(1,N);

% for each node, initialize the factors, find the neighbors, store it in neighbor(N, K)
for i=1:N
    out_host(i,:)=rand(1,dim); % in-factors
    in_host(:,i)=rand(dim,1);  % out-factors

    % Randomly selected Neighbors
      tmp = randperm(N);
      
      point = 1;
      for j=1:K
          neighbor(i, j) = -1;
      end
      for j=1:K
          % fill in neighbor(i, j) %
          if (i ~= tmp(point) && D_change(i, tmp(point)) > 0) % D_change: the distance matrix
              neighbor(i, j) = tmp(point);
              point = point + 1;
          else
             while(i == tmp(point) || D_change(i, tmp(point)) <= 0) % || D_change(i, tmp(point)) < 0)
                point = point + 1;
                if (point > N)
                    break;
                end
             end      
          end
              if (point > N)
                  break;
              end              
      end
      

end

% Out : X in the paper, In : Y in the paprt
%delta_out = delta;
%delta_in = delta;

%for j=1:15

w=zeros(N, K, dim);
h=zeros(N, dim, K);
D_host2landmark=zeros(N, K);
D_host2landmark_out=zeros(N, K);
D_host2landmark_in=zeros(N, K);

%for j = 1:100
%for j = 1:(iteration/2)
fpre_newhost = [];
fpre_flashcrowd = [];

% 2011-02-28, host churn %

host_age = zeros(1, N); %age of each host, -1 means the dead ones
host_ttl = zeros(1, N); % how much time does a host have? only meaningful for alive hosts
host_ttj = zeros(1, N); % how much time does a dead host have for re-joining?


start_time = 0; Init_alive_percentage = 0.8;

for i=1:N
    if (rand < Init_alive_percentage)
        % node i is alive
        host_age(i) = 0;
        %host_ttl(i) = uptime_generator(avg_up_length);
        host_ttl(i) = pareto_dist(mean_on_length/2, 1.5);
        %host_ttl(i) = ceil(-100*log(1-rand));
        host_ttj(i) = -1;
    else
        % node i is dying
        host_age(i) = -1;
        host_ttl(i) = -1;
        host_ttj(i) = ceil(-120*log(1-rand));
    end
end

current_time = 0;
time_bound = 3600*3;
evolution_npre = [];

for round = 1:time_bound;
    %current_time = current_time + round_length;
    alive_host = find(host_age>=0);dead_host = find(host_age<0);
    host_age(alive_host) = host_age(alive_host) + 1;% process host_age %
    host_ttl(alive_host) = host_ttl(alive_host) - 1;% process host_ttl, time-to-live
    host_ttj(dead_host) = host_ttj(dead_host) - 1; % process host_ttj, time-to-?
    new_born_host_list = find(host_age(i) == 1);% Justify new hosts
     
%     if (round == 1)
%         % First round : Joining %        
%         % NMF for first K nodes %
%         D_landmark = D(1:K, 1:K);
%         [out_landmark, in_landmark] = NMF(D_landmark, dim);
%         %%%%%%%%%%%%%%%%%%%%%%%%%
%         out_host(1:K, :) = out_landmark;
%         in_host(:, 1:K) = in_landmark;
%         % Setup neighbors %
%         for i=1:N
%             if (i<=K)
%                 reference_hosts(i, :) = 1:K;
%             else
%                 rand_existing = randperm(i-1);
%                 reference_hosts(i, :) = rand_existing(1:K);
%             end
%         end             
% 
%     else
%         % Setup neighbors %
%         for i=1:N
%             if (i<=K)
%                 reference_hosts(i, :) = 1:K;
%             else
%                 rand_existing = randperm(i-1);
%                 reference_hosts(i, :) = rand_existing(1:K);
%             end
%         end   
%        
%     end
    
    for i = 1:N
        if (host_age(i) == -1)
            continue;
        end
        if (mod(host_age(i), round_length) ~= 0)
            continue;
        end
        % K: number of landmarks
        % w: Kxd, h: dxK. The position vectors of all landmarks
       
        if (host_age(i) == 0) % Initial NC Calculation
            out_host(i,:)=rand(1,dim);
            in_host(:,i)=rand(dim,1);
            init_coord = 1;
            if (init_coord == 1)
                target_host = neighbor(i, :);
                target_host = target_host(find(target_host>0));
                target_host = target_host(find(D(i, target_host)>=0));
            
                weight_out_vec = zeros(1, length(target_host)); % K -> actual_K
                weight_in_vec = zeros(1, length(target_host)); % K -> actual_K

                temp_w = out_host(target_host,:);
                temp_h = in_host(:, target_host);
                temp_D_host2landmark = D_change(i, target_host);
                 temp_D_host2landmark_out = temp_D_host2landmark(1:length(target_host));
                 temp_D_host2landmark_in = temp_D_host2landmark(1:length(target_host));

                for ii=1:1:length(target_host)
                    if (D(i,  target_host(ii)) < 0)
                        weight_out_vec(ii) = eps;             
                        weight_in_vec(ii) = eps;           
                    else
                        weight_out_vec(ii) = 1;             
                        weight_in_vec(ii) = 1;           
                    end
                end
                t = weight_lsqnonneg(temp_h', temp_D_host2landmark_in', sqrt(weight_in_vec)');
                out_host(i, :) = t';        
                in_host(:, i) = weight_lsqnonneg(temp_w, temp_D_host2landmark_out', sqrt(weight_out_vec)');
            end
        end
             
        % target 
        
        target_host = neighbor(i, :);        
        target_host = target_host(find(target_host>0));        
        
        actual_K = length(target_host);        
        temp_w = out_host(target_host,:);
        temp_h = in_host(:, target_host);
        temp_D_host2landmark = D_change(i, target_host);
        


        % get the score of all hosts in neighbor(i, :) %
        score_out_vec = zeros(1, actual_K); % K -> actual_K
        score_in_vec = zeros(1, actual_K); % K -> actual_K
        score_aver_vec = zeros(1, actual_K); % K -> actual_K
        
        weight_out_vec = zeros(1, actual_K); % K -> actual_K
        weight_in_vec = zeros(1, actual_K); % K -> actual_K

        
        for index_nb=1:actual_K % K -> actual_K
            predict_ii_in = temp_w(index_nb, :) * in_host(:, i);
            predict_ii_out = out_host(i, :) * temp_h(:, index_nb);
            s1 = abs(predict_ii_out - D_change(i, target_host(index_nb)));% / (D_change(i, neighbor(i, index_nb))+eps);
            s2 = abs(predict_ii_in - D_change(i, target_host(index_nb)));% / (D_change(i, neighbor(i, index_nb))+eps);
            score_out_vec(index_nb) = s1;
            score_in_vec(index_nb) = s2;
        end
        
        out_threshold = median(score_out_vec);
        in_threshold = median(score_in_vec);


        for ii=1:actual_K % K -> actual_K
            if (score_out_vec(ii) <= out_threshold)
                weight_out_vec(ii) = 1;
            else
                if (score_out_vec(ii) < out_threshold * C)% || score_out_vec(ii)  < out_upbound_threshold)
                    weight_out_vec(ii) = (out_threshold/score_out_vec(ii))^2;           
                else
                    weight_out_vec(ii) = eps;
                end            
            end
            if (score_in_vec(ii) <= in_threshold)
                weight_in_vec(ii) = 1;
            else
                if (score_in_vec(ii) < in_threshold * C)% || score_in_vec(ii)  < in_upbound_threshold)
                    weight_in_vec(ii) = (in_threshold/score_in_vec(ii))^2;           
%                    weight_in_vec(ii) = 1 - score_in_vec(ii);
                else
                    weight_in_vec(ii) = eps;%1/C;
                end 
            end
        end
         
         temp_D_host2landmark_out = temp_D_host2landmark(1:actual_K); % K -> actual_K
         temp_D_host2landmark_in = temp_D_host2landmark(1:actual_K); % K -> actual_K

%        t = lsqnonneg(temp_h', temp_D_host2landmark_in');
        t = weight_lsqnonneg(temp_h', temp_D_host2landmark_in', sqrt(weight_in_vec)');
        out_host(i, :) = t';
        
        %  Out_NCs[K*d] * in_host(:, i) [d*1] = D_host2landmark'[K*1];
        %  => w * in_host(:, i) = D_hosts2landmark'[K*1];
        
%        in_host(:, i) = lsqnonneg(temp_w, temp_D_host2landmark_out');
        in_host(:, i) = weight_lsqnonneg(temp_w, temp_D_host2landmark_out', sqrt(weight_out_vec)');

%         w = backup_w;
%         h = backup_h;
%         D_host2landmark = backup_D_host2landmark;
    
    end;
    
    % process host_ttl
    alive_host_list = find(host_ttl > 0);
    alive_num = length(alive_host_list);
    %fprintf('[%d]', length(alive_host_list));
    
    if (mod(round, 100) == 0 & round > 9300) % 3600 * 3 - 1500
        fprintf('[%d]', length(alive_host));
        measured_matrix = D(alive_host_list, alive_host_list);
        predicted_matrix = out_host(alive_host_list, :) * in_host(:, alive_host_list);
        rerr=relative_error(predicted_matrix, measured_matrix);
        npre_current = NPRE(rerr);
        fprintf('%.3f ', npre_current);
        evolution_npre = [evolution_npre npre_current];        
    end
%     if (mod(round, 1000) == 0)
%         fprintf('<%d> ', round);
%     end
    
    whole_round_related = zeros(1, N);
    % find the related hosts first...
    for i=1:N
        if (host_ttl(i) == 0) % alive hosts -> death
            related_host_list = [];
            [xx, yy]=find(neighbor==i);
            whole_round_related(xx) = 1;
        end
    end
    for i=1:N
        if (whole_round_related(i) == 1) % check if alive
            whole_round_related(i) = ismember(i, alive_host_list);
        end
    end
    whole_round_related_num = sum(whole_round_related);
    whole_round_related_set = find(whole_round_related==1);
    %fprintf('\n%d: ', whole_round_related_num);
    %tic
    two_hop_matrix = zeros(N, N);
    for count=1:whole_round_related_num
        i=whole_round_related_set(count);
        current_nb_vec_init = neighbor(i, :);
        current_nb_vec = current_nb_vec_init(find(current_nb_vec_init>0));
        current_2hop_matrix = neighbor(current_nb_vec, :);
        current_2hop_vec = current_2hop_matrix(:);
        two_hop_matrix(i, current_2hop_vec(find(current_2hop_vec>0))) = 1;
        
    end
    %toc
    for i=1:N
        if (host_ttl(i) == 0) % alive hosts -> death
            host_ttl(i) = -1;
            host_age(i) = -1;
            host_ttj(i) = ceil(-120*log(1-rand));           
            [xx, yy] = find(neighbor == i);
            
            % start the replacement...
            replacement_num = length(xx);
            for ii=1:replacement_num
                if (ismember(xx(ii), alive_host_list))
                    neighbor(xx(ii), yy(ii)) = -1;                
                    two_hop_vec = find(two_hop_matrix(xx(ii), :)==1);
                    two_hop_alive = intersect(two_hop_vec, alive_host_list);
                    two_hop_alive_nodup = setdiff(two_hop_alive, neighbor(xx(ii), :));
                    if (length(two_hop_alive_nodup) == 0)
                        two_hop_alive_nodup = setdiff(alive_host_list, neighbor(xx(ii), :));                        
                    end
%                    fprintf('(%d,%d) ', length(two_hop_alive), length(two_hop_alive_nodup));
                    neighbor(xx(ii), yy(ii)) = two_hop_alive_nodup(ceil(rand*length(two_hop_alive_nodup)));
                end
            end
            for ii=1:K
                neighbor(i, ii) = -1;
            end
        end
        if (host_ttj(i) == 0) % dead hosts -> alive, fetch a random list from RP
            host_ttj(i) = -1;
            host_age(i) = 0;
            host_ttl(i) = pareto_dist(mean_on_length/2, 1.5);
            %host_ttl(i) = ceil(-100*log(1-rand));
            % create the neighbor list, selected from the alive hosts...
            tmp_seq=randperm(length(alive_host_list));
            if (length(alive_host_list) < K)
                'warning'
                alive_host_list
            end
            neighbor(i, :) = alive_host_list(tmp_seq(1:K));
        end
    end
    
end;
tlength = length(evolution_npre);
stabilized_npre = median(evolution_npre((tlength-10):tlength));
fprintf('\nRound Length=%d, NPRE = %.3f\n', round_length, stabilized_npre);
