function W = select_neighbors(M, K)

n = size(M, 1);
W = false(n);
for i = 1:n
    valid = find(M(i,:) > 0);
    list = randperm(length(valid));
    if length(list) >= K
        list = list(1:K);
    else
        warning('MATLAB:w1', 'not enougth neighors too choose from: K = %d, links = %d\n', K, length(list));
    end
    W(i, valid(list)) = true;
end

end