
function log_odds = gnb_predict(X, theta, mu_1, mu_0, sd)
    m = size(X, 1);
    p_x_y_1 = normpdf(X, mu_1, sd);
    p_x_y_0 = normpdf(X, mu_0, sd);
    log_odds = log(theta/(1-theta)) + sum(log(p_x_y_1./p_x_y_0),2);
end
