% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 2, due October 26

function [log_odds, p_x_y_1, p_x_y_0] = ...
                gnb_predict(X, theta, mu_1, mu_0, Sigma)
%% [log_odds, p_x_y_1, p_x_y_0] = gnb_predict(X, theta, mu_1, mu_0, Sigma)

% Calculates the log-odds ratio of aseries of inputs pertaining to class
% 0 or class 1. X is the matrix of observations and THETA, MU_1, MU_0,
% SIGMA are the model parameters. In addition to returning the log-odds
% ratio, it also returns the estimated joint probability of each
% observation pertaining to class 0 or class 1 respectively.

    % amount of observations
    m = size(X, 1);
    % size of feature vector
    n = size(X, 2);
    Sigma_inv = pinv(Sigma);
    norm_term = 1/((2*pi)^(n/2)*sqrt(det(Sigma)));
    X_to_mu_1 = (X - repmat(mu_1, m, 1));
    X_to_mu_0 = (X - repmat(mu_0, m, 1));
    % essentially, running (x_i - mu_c)^{T} * Sigma^{-1} * (x_i - mu_c)
    % will result in a scalar that is equivalent ot the corresponding
    % diagonal entry in the vectorized (quicker) version. 
    % .*eye(m))*ones(m,1) will keep a column with the diagonal entries of
    % an m x m matrix...
    alpha_1 = ((X_to_mu_1 * Sigma_inv * X_to_mu_1').*eye(m))*ones(m,1);
    alpha_0 = ((X_to_mu_0 * Sigma_inv * X_to_mu_0').*eye(m))*ones(m,1);
    p_x_y_1 = norm_term*exp(-alpha_1./2);
    p_x_y_0 = norm_term*exp(-alpha_0./2);
    log_odds = (alpha_0./2 - alpha_1./2) + ...
                repmat(log(theta/(1 - theta)), length(alpha_1), 1);
end
