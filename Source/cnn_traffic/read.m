path='D:\Test\2011NHNX_speed';
target='D:\Test\2011NHNX_speed_fix';
a=dir(fullfile(path,'*.mat'));
liuliangshuju=ones(35,288);
for  i=1:length(a);
    load(fullfile(path,a(i).name));
    sudushuju=speed;
    for j=1:72
        for k=1:288
            if isnan(sudushuju(j,k))
                sudushuju(j,k)=50+20*rand-10;
            end
            if (sudushuju(j,k)==1||sudushuju(j,k)==0)&&j~=1&&j~=72
               % if j==1
                %    sudushuju(j,k)=sudushuju(j+1,k);
                %end
                %if j==35
                 %   sudushuju(j,k)=sudushuju(j-1,k);
                %end
                if sudushuju(j,k)==1
                    if sudushuju(j+1,k)>=1&&sudushuju(j-1,k)>=0
                        sudushuju(j,k)=(sudushuju(j+1,k)+sudushuju(j-1,k))/2;
                    end
                end
                if sudushuju(j,k)==0
                    if sudushuju(j+1,k)>=1&&sudushuju(j-1,k)>=0
                        sudushuju(j,k)=(sudushuju(j+1,k)+sudushuju(j-1,k))/2;
                    end
                end
            %elseif sudushuju(j,k)==0
            %    sudushuju(j,k)=50+20*rand-10;
            end
            if sudushuju(j,k)==0||sudushuju(j,k)==1
                sudushuju(j,k)=50+20*rand-10;
            end
        end
    end
    for j=1:72
        for k=1:288
            if isnan(sudushuju(j,k))
                sudushuju(j,k)=50+20*rand;
            end
        end
    end
    if i<10
        save (fullfile(target,['00',num2str(i),'.mat']), 'sudushuju')
    end
    if i>=10&&i<100
        save (fullfile(target,['0',num2str(i),'.mat']),'sudushuju')
    end
    if i>=100
        save(fullfile(target,[num2str(i),'.mat']),'sudushuju')
    end            
end
