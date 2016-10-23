
function [theta, mu_1, mu_0, sd] = gnb_train(X, y)
    theta = sum(y == 1)/length(y);
    idx_1 = y == 1;
    idx_0 = y == 0;
    mu_1 = mean(X(idx_1,:));
    mu_0 = mean(X(idx_0,:));
    sd = std(X);
end