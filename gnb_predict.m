
function [log_odds, p_x_y_1, p_x_y_0] = ...
                gnb_predict(X, theta, mu_1, mu_0, Sigma)
    m = size(X, 1);
    n = size(X, 2);
    Sigma_inv = pinv(Sigma);
    norm_term = 1/((2*pi)^(n/2)*sqrt(trace(Sigma)));
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
    log_odds = (alpha_1 ./ alpha_0) + ...
                repmat(log(theta/(1 - theta)), length(alpha_1), 1);
end
