path='C:\Users\Tianyang\Desktop\深度学习的毕业设计\数据处理速度mat';
a=dir(fullfile(path,'*.mat'));
liuliangshuju=ones(35,288);
for  i=1:length(a);
    load(fullfile(path,a(i).name));
    sudushuju=sudu;
    for j=1:35
        for k=1:288
            if isnan(sudushuju(j,k))
                sudushuju(j,k)=50+20*rand;
            end
            if sudushuju(j,k)==1&&j~=1&&j~=35
               % if j==1
                %    sudushuju(j,k)=sudushuju(j+1,k);
                %end
                %if j==35
                 %   sudushuju(j,k)=sudushuju(j-1,k);
                %end
                if sudushuju(j,k)==1
                    sudushuju(j,k)=(sudushuju(j+1,k)+sudushuju(j-1,k))/2;
                end
            end            
        end
    end
    for j=1:35
        for k=1:288
            if isnan(sudushuju(j,k))
                sudushuju(j,k)=50+20*rand;
            end
        end
    end
    if i<10
        save (['00',num2str(i),'.mat'], 'sudushuju')
    end
    if i>=10&&i<100
        save (['0',num2str(i),'.mat'] ,'sudushuju')
    end
    if i>=100
        save([num2str(i),'.mat'],'sudushuju')
    end            
end
