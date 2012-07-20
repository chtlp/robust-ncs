function plot_relative_error(estimate, real, color)
    rerr=relative_error(estimate, real);
    plot_cumu_matrix(rerr, color);
    xlim([0 1]);
    %xlim([-1 1]);
    hold on
