function m = medianAbsoluteError(X,Xhat)

ids = find(X);

x = X(ids);
xhat = Xhat(ids);

m = median(abs(x-xhat));

return