function rerr = relativeError(X, Xhat)

ids = find(X>0);

x = X(ids);
xhat = Xhat(ids);

rerr = abs(x-xhat)./x;
