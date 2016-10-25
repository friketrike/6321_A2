f3h = figure(3);
set(f3h, 'Position',[10 200 480 480]);
f4h = figure(4);
set(f4h, 'Position',[560 200 480 480]);
for i = 1:32
    figure(3)
    plot(X(y==1, i), zeros(sum(y==1),1), 'g.')
    hold on
    plot(X(y==0, i), zeros(sum(y==0),1), 'r.')
    exes = min(X(:,i))-1:0.1:max(X(:,i))+1;
    plot(exes, normpdf(exes, mean(X(y == 1, i)), std(X(y == 1, i))), 'g')
    plot(exes, normpdf(exes, mean(X(y == 0, i)), std(X(y == 0, i))), 'r')
    title(['feature ', num2str(i), ' class distributions'])
    hold off
    for j = i+1:32
        figure(4)
        plot(X((y == 1),i), X((y == 1),j), 'g+')
        hold on
        plot(X((y == 0),i),X((y == 0),j), 'r.')
        hold off
        title(['feature ', num2str(i), ' against feature ', num2str(j)]);
        pause(0.1)
    end
end