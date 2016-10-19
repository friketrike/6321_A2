function alpha = find_alpha(X, y, w)
    lrs = 1./(1:10);
    errors = zeros(1, length(lrs));
    for i = 1: length(lrs)
        iterations = 1;
        lr = lrs(i);
        while iterations < 100
            epsilon = y - (1./(1+exp(-X*w)));
            % TODO check...
            % lr = 1/(1+ iterations);
            w = w + lr * (X'*epsilon);
            iterations = iterations + 1;
        end
        errors(i) = sum(epsilon);
    end
    [dummy, idx] = min(abs(errors));
    alpha = lrs(idx);
end
