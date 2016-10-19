function [ w, w_inits ] = LR_grad( X, y, w_inits, num_rand_inits )
%LR_GRAD Summary of this function goes here
%   Detailed explanation goes here

    % default to 4 random starting points
    if nargin < 4 || isempty(num_rand_inits)
       num_rand_inits = 4; 
    end

    % if no random starting points are 
    if nargin < 3 || isempty(w_inits)
        w_inits = 10*rand(size(X,2), num_rand_inits); 
        %TODO, see how far apart these random points should be
        w_inits(:,1) = 0; % TODO remove...
    else
        num_rand_inits = size(w_inits(2));
    end
    
    w = w_inits;
    for i = 1: num_rand_inits
        done = false;
        prev_epsilon = realmax * ones(length(y),1);
        iterations = 1;
        lr = find_alpha(X, y, w(:,i));
        while ~done && (iterations < 10000)
            epsilon = y - (1./(1+exp(-X*w(:,i))));
            % TODO, think about cutoff strategy, observe behaviour...
            if (abs(sum(epsilon)) < .05) || abs(sum(epsilon)-sum(prev_epsilon)) < 0.005
                done = true;
            else
                w(:,i) = w(:,i) + lr * (X'*epsilon);
                prev_epsilon = epsilon;
            end
            iterations = iterations + 1;
        end
    end
end

