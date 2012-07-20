function plot_cumu_matrix(m,str, precision)

if nargin<3 precision=1000; end
if nargin<2 str=''; end

sorted = sort(m(:));
N = length(sorted);
m = N;
if N>precision m=precision; end
m_array = 1/m:1/m:1;
pos_array = round(m_array.*N);
plot(sorted(pos_array), m_array, str);

return
%plot a cumulative graph of matrix m with M*N
    [M,N] = size(m);
    a = M*N;
    m_array = 1/a:1/a:1;
    f_array = reshape(m,1,a);
    plot(sort(f_array),m_array,str)

