function alpha = find_alpha(X, y, w)
%% For a given problem defined by X and Y
% as well as a starting point W for gradient 
% descent, this function returns an optimal
% learning rate from {1, 1/2, ... 1/10}.
    lrs = 1./(1:10);
    errors = zeros(1, length(lrs));
    for i = 1: length(lrs)
        iterations = 1;
        lr = lrs(i);
        while iterations < 100
            epsilon = y - (1./(1+exp(-X*w)));
            w = w + lr * (X'*epsilon);
            iterations = iterations + 1;
        end
        errors(i) = sum(epsilon);
    end
    [dummy, idx] = min(abs(errors));
    alpha = lrs(idx);
end
