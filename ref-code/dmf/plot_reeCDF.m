function [ycdf,xcdf] = plot_reeCDF(X, Xhat, LineSpec)

rerr = relativeError(X,Xhat);

[ycdf,xcdf] = cdfcalc(rerr);
ycdf = ycdf(2:length(ycdf));

plot(xcdf,ycdf,LineSpec,'LineWidth',2);

return 