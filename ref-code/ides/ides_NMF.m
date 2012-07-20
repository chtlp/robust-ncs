function [out_landmark, in_landmark, out_host, in_host] = ides_NMF(D_landmark, D_host2landmark, dim)
% [out_landmark, in_landmark, out_host, in_host] = ides(D_landmark, D_host2landmark, dim)
% D_landmark: LxL pairwise distance matrix of landmarks. L is the
% number of landmarks
% D_host2landmark: NxL distance matrix where N is the number of
% ordinary hosts
% dim: in/out vector dimensions
% output: 
%   out_landmark: outgoing vectors of landmarks
%   in_landmark: incoming vectors of landmarks
%   out_host: outgoing vectors of ordinary hosts
%   in_host: incoming vectors of ordinary hosts

[out_landmark, in_landmark] = NMF(D_landmark, dim);
[out_host, in_host] = newhosts_NMF(out_landmark, in_landmark, ...
				   D_host2landmark, D_host2landmark', ...
				   1:length(D_landmark));
