speed_path='D:\MasterDL\data_set\2011南北东线速度数据';
volume_path='D:\MasterDL\data_set\2011南北东线流量数据';
target_path='D:\Test\2011NBDX_speed';
af=dir(fullfile(speed_path,'*.csv'));
bf=dir(fullfile(volume_path,'*.csv'));
v=200;
for  k=1:length(af)
    %load(fullfile(path,a(i).name));
    [g, txts, raw] = xlsread(fullfile(speed_path,af(k).name));
    [r, txtv, raw2] = xlsread(fullfile(volume_path,bf(k).name));
    
    d=zeros(43,288);
r(1,:)=[];
raw(1,:)=[];
%编号0-9的检测器
s=size(r);
l=s(1,1);
for i=1:9
    num=0;
    a=i;
    
    for j=1:l
        f=raw{j,1}(5:6);
    if a==0  
        m=strfind(f,'00');
    elseif a==1
        m=strfind(f,'01');
    elseif a==2
        m=strfind(f,'02');
    elseif a==3
        m=strfind(f,'03');
    elseif a==4
        m=strfind(f,'04');
    elseif a==5
        m=strfind(f,'05');
    elseif a==6
        m=strfind(f,'06');
    elseif a==7
        m=strfind(f,'07');
    elseif a==8
        m=strfind(f,'08');
    else
        m=strfind(f,'09');
    end
        if isempty(m)==0
            num=num+1;
            for n=1:288
                d(i,n)=d(i,n)+r(j,n);
            end
        end
    end
   
end
%编号10-34的检测器
for i=10:43
    num=0;
    a=i;
    
    for j=1:l
        f=raw{j,1};
    if a==10
        m=strfind(f,'10');
    elseif a==11
        m=strfind(f,'11');
    elseif a==12
        m=strfind(f,'12');
    elseif a==13
        m=strfind(f,'13');
    elseif a==14
        m=strfind(f,'14');
    elseif a==15
        m=strfind(f,'15');
    elseif a==16
        m=strfind(f,'16');
    elseif a==17
        m=strfind(f,'17');
    elseif a==18
        m=strfind(f,'18');
    elseif a==19
        m=strfind(f,'19');
    elseif a==20
        m=strfind(f,'20');
    elseif a==21
        m=strfind(f,'21');
    elseif a==22
        m=strfind(f,'22');
    elseif a==23
        m=strfind(f,'23');
    elseif a==24
        m=strfind(f,'24');
    elseif a==25
        m=strfind(f,'25');
    elseif a==26
        m=strfind(f,'26');
    elseif a==27
        m=strfind(f,'27');
    elseif a==28
        m=strfind(f,'28');
    elseif a==29
        m=strfind(f,'29');
    elseif a==30
        m=strfind(f,'30');
    elseif a==31
        m=strfind(f,'31');
    elseif a==32
        m=strfind(f,'32');
    elseif a==33
        m=strfind(f,'33');
    elseif a==34
        m=strfind(f,'34');
    elseif a==35
        m=strfind(f,'35');
    elseif a==36
        m=strfind(f,'36');
    elseif a==37
        m=strfind(f,'37');
    elseif a==38
        m=strfind(f,'38'); 
    elseif a==39
        m=strfind(f,'39');
    elseif a==40
        m=strfind(f,'40');
    elseif a==41
        m=strfind(f,'41');
    elseif a==42
        m=strfind(f,'42');
    else
        m=strfind(f,'43');
    end
        if isempty(m)==0
            num=num+1;
            for n=1:288
                d(i,n)=d(i,n)+r(j,n);
            end
        end
    end
    
end  
d2=zeros(43,288);
g(1,:)=[];
raw2(1,:)=[];
%编号0-9的检测器
s=size(g);
l=s(1,1);
for i=1:9
    num=0;
    a=i;
    for j=1:l
        f=raw2{j,1}(5:6);
    if a==0  
        m=strfind(f,'00');
    elseif a==1
        m=strfind(f,'01');
    elseif a==2
        m=strfind(f,'02');
    elseif a==3
        m=strfind(f,'03');
    elseif a==4
        m=strfind(f,'04');
    elseif a==5
        m=strfind(f,'05');
    elseif a==6
        m=strfind(f,'06');
    elseif a==7
        m=strfind(f,'07');
    elseif a==8
        m=strfind(f,'08');
    else
        m=strfind(f,'09');
    end
        if isempty(m)==0
            num=num+1;
            for n=1:288
                d2(i,n)=d2(i,n)+g(j,n)*r(j,n);
            end
        end
    end
   for n=1:288
       if d(i,n)==0
           d2(i,n)=0;%识别无效数据
       else
           d2(i,n)=d2(i,n)/d(i,n);
       end
   end
end
%编号10-34的检测器
for i=10:43
    num=0;
    a=i;
    
    for j=1:l
        f=raw2{j,1};
    if a==10
        m=strfind(f,'10');
    elseif a==11
        m=strfind(f,'11');
    elseif a==12
        m=strfind(f,'12');
    elseif a==13
        m=strfind(f,'13');
    elseif a==14
        m=strfind(f,'14');
    elseif a==15
        m=strfind(f,'15');
    elseif a==16
        m=strfind(f,'16');
    elseif a==17
        m=strfind(f,'17');
    elseif a==18
        m=strfind(f,'18');
    elseif a==19
        m=strfind(f,'19');
    elseif a==20
        m=strfind(f,'20');
    elseif a==21
        m=strfind(f,'21');
    elseif a==22
        m=strfind(f,'22');
    elseif a==23
        m=strfind(f,'23');
    elseif a==24
        m=strfind(f,'24');
    elseif a==25
        m=strfind(f,'25');
    elseif a==26
        m=strfind(f,'26');
    elseif a==27
        m=strfind(f,'27');
    elseif a==28
        m=strfind(f,'28');
    elseif a==29
        m=strfind(f,'29');
    elseif a==30
        m=strfind(f,'30');
    elseif a==31
        m=strfind(f,'31');
    elseif a==32
        m=strfind(f,'32');
    elseif a==33
        m=strfind(f,'33');
    elseif a==34
        m=strfind(f,'34');
    elseif a==35
        m=strfind(f,'35');
    elseif a==36
        m=strfind(f,'36');
    elseif a==37
        m=strfind(f,'37');
    elseif a==38
        m=strfind(f,'38'); 
    elseif a==39
        m=strfind(f,'39');
    elseif a==40
        m=strfind(f,'40');
    elseif a==41
        m=strfind(f,'41');
    elseif a==42
        m=strfind(f,'42');
    else
        m=strfind(f,'43');
    end
        if isempty(m)==0
            num=num+1;
            for n=1:288
                d2(i,n)=d2(i,n)+g(j,n)*r(j,n);
            end
        end
    end
    for n=1:288
       if d(i,n)==0
           d2(i,n)=0;%识别无效数据
       else
           d2(i,n)=d2(i,n)/d(i,n);
       end
   end
end

for i=2:43
    sum=0;
    for j=1:288
        sum=sum+d(i,j);
    end
    if sum==0
        for n=1:288
            d(i,n)=(d(i+1,n)+d(i-1,n))/2;
            %d1(i,n)=(d1(i+1,n)+d1(i-1,n))/2;
            d2(i,n)=(d2(i+1,n)+d2(i-1,n))/2;
        end
    end
end

%速度异常值
for i=1:43
    for j=2:288
        if d2(i,j)>v
            d2(i,j)=(d2(i,j-1)+d2(i,j+1))/2;
        end
    end
end

volume=flipud(d);
speed=flipud(d2);
    if k<10
        save (fullfile(target_path,['00',num2str(k),'.mat']), 'speed')
    end
    if k>=10&&k<100
        save (fullfile(target_path,['0',num2str(k),'.mat']),'speed')
    end
    if k>=100
        save(fullfile(target_path,[num2str(k),'.mat']),'speed')
    end
end