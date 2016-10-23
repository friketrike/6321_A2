for i = 1:32
    for j = i:32
        plot(X((y == 1),[i,j]), 'g+')
        hold on
        plot(X((y == 0),[i,j]), 'r.')
        hold off
        title(['feature ', num2str(i), ' against feature ', num2str(j)]);
        pause(0.1)
    end
end