
function [theta, mu_1, mu_0, Sigma] = gnb_train(X, y)
    theta = sum(y == 1)/length(y);
    idx_1 = y == 1;
    idx_0 = y == 0;
    mu_1 = mean(X(idx_1,:));
    mu_0 = mean(X(idx_0,:));
    x_to_mu = zeros(size(X));
    x_to_mu(idx_1,:) = X(idx_1,:) - repmat(mu_1, sum(idx_1), 1);
    x_to_mu(idx_0,:) = X(idx_0,:) - repmat(mu_1, sum(idx_0), 1);
    Sigma = (x_to_mu' * x_to_mu)./length(y);
end