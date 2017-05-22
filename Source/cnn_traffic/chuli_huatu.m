clc
clear
[r,txt,raw]=xlsread('C:\Users\ASUS-A550J\Desktop\流量、速度、占有率数据\2011q\2011q (1).csv');
v=200;
d=zeros(35,288);
r(1,:)=[];
raw(1,:)=[];
%编号0-9的检测器
s=size(r);
l=s(1,1);
for i=1:10
    num=0;
    a=i-1;
    
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
for i=11:35
    num=0;
    a=i-1;
    
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
    else
        m=strfind(f,'34');
    end
        if isempty(m)==0
            num=num+1;
            for n=1:288
                d(i,n)=d(i,n)+r(j,n);
            end
        end
    end
    
end     
%检测器处理
d2=zeros(35,288);
g(1,:)=[];
raw2(1,:)=[];
%编号0-9的检测器
s=size(g);
l=s(1,1);
for i=1:10
    num=0;
    a=i-1;
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
for i=11:35
    num=0;
    a=i-1;
    
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
    else
        m=strfind(f,'34');
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
%检测器处理
d1=zeros(35,288);
b(1,:)=[];
raw3(1,:)=[];
%编号0-9的检测器
s=size(b);
l=s(1,1);
for i=1:10
    num=0;
    a=i-1;
    
    for j=1:l
        f=raw3{j,1}(5:6);
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
                d1(i,n)=d1(i,n)+b(j,n);
            end
        end
    end
   for n=1:288
       d1(i,n)=d1(i,n)/num;
   end
end
%编号10-34的检测器
for i=11:35
    num=0;
    a=i-1;
    
    for j=1:l
        f=raw3{j,1};
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
    else
        m=strfind(f,'34');
    end
        if isempty(m)==0
            num=num+1;
            for n=1:288
                d1(i,n)=d1(i,n)+b(j,n);
            end
        end
    end
    for n=1:288
       d1(i,n)=d1(i,n)/num;
    end
end     
%检测器缺失
for i=2:35
    sum=0;
    for j=1:288
        sum=sum+d(i,j);
    end
    if sum==0
        for n=1:288
            d(i,n)=(d(i+1,n)+d(i-1,n))/2;
            d1(i,n)=(d1(i+1,n)+d1(i-1,n))/2;
            d2(i,n)=(d2(i+1,n)+d2(i-1,n))/2;
        end
    end
end

%速度异常值
for i=1:35
    for j=2:288
        if d2(i,j)>v
            d2(i,j)=(d2(i,j-1)+d2(i,j+1))/2;
        end
    end
end
%倒序
d=flipud(d);
d1=flipud(d1);
d2=flipud(d2);
%d-流量 d2-速度 d1-占有率
%画图
h1=max(d);
h1=max(h1);
h2=max(d2);
h2=max(h2);
h3=100;
d=d/h1;
d2=d2/h2;
d1=d1/h3;
x1=1:1:288;
y1 = 0:1:34;
set(gcf,'unit','centimeters','position',[10 5 18 15]);
rgb=cat(3,d,d2,d1);
imagesc(rgb)
set(gca,'XTick',[0:12:288]);  
set(gca,'XTickLabel',{'  0:00','  1:00', '  2:00', '  3:00' ,'  4:00', '  5:00','  6:00','  7:00','  8:00', '  9:00','  10:00','  11:00','  12:00','  13:00','  14:00','  15:00','  16:00','  17:00','  18:00','  19:00','  20:00','  21:00','  22:00', '  23:00','  24:00'});
set(gca,'YTick',[1:1:35]);  
set(gca,'YTickLabel',{'YABX34','YABX33', 'YABX32', 'YABX31' ,'YABX30', 'YABX29','YABX28','YABX27','YABX26', 'YABX25','YABX24','YABX23','YABX22','YABX21','YABX20','YABX19','YABX18','YABX17','YABX16','YABX15','YABX14','YABX13','YABX12','YABX11','YABX10','YABX09','YABX08','YABX07','YABX06','YABX05','YABX04','YABX03','YABX02','YABX01','YABX00'});

he=gca;
rot=90;
a=get(he,'XTickLabel');
set(he,'XTickLabel',[]);
b=get(he,'XTick');
c=get(he,'YTick');
th=text(b,repmat(c(1)-.1*(c(2)-c(1)),length(b),1),a,'HorizontalAlignment','left','rotation',rot,'FontSize',10);






