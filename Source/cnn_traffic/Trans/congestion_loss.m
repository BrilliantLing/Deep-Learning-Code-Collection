mre_list = [];
for i=1:35
    for j = 1:108
        if b(i,j)<20
            mre = abs(a(i,j)-b(i,j))/b(i,j);
            mre_list = [mre_list, mre];
            if mre > 0.5
                b(i,j)
                a(i,j)
            end
        end
    end
end

mre_list;
mean_relative_error = mean(mre_list)