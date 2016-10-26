% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 2, due October 26

function [theta, mu_1, mu_0, Sigma] = gnb_train(X, y)
%% [theta, mu_1, mu_0, Sigma] = gnb_train(X, y)

% Returns the necessary parameters for binary classification
% of continuous inputs, where X is the observations matrix and 
% Y is the associated label vector.  

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
