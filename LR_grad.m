function [ w, w_inits ] = LR_grad( X, y, lr, w_inits, num_rand_inits )
%LR_GRAD Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 3 || isempty(lr)
       lr = 0.5;
       % TODO, something smarter here to find a better learning rate
    end

    % default to 4 random starting points
    if nargin < 5 || isempty(num_rand_inits)
       num_rand_inits = 4; 
    end

    % if no random starting points are 
    if nargin < 4 || isempty(w_inits)
        w_inits = 10*rand(size(X,2), num_rand_inits); 
        %TODO, see how far apart these random points should be
    else
        num_rand_inits = size(w_inits(2));
    end

    for i = 1: num_rand_inits
        w = w_inits(:,i);
        done = false;
        prev_epsilon_mean = realmax;
        while ~done
            epsilon = y - (1./(1+exp(-X*w)));
            %i
            %mean(epsilon)
            w = w + lr * (X'*epsilon);
            % TODO, think about cutoff strategy, observe behaviour...
            if (abs(prev_epsilon_mean - mean(epsilon)) < .0000001)
                done = true;
            else
                prev_epsilon_mean = mean(epsilon);
            end
        end
    end
end

