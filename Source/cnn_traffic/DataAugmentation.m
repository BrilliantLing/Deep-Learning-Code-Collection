data_path = 'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\tomorrow\';
augment_path = 'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\tomorrow_augment\';
data_files = dir(fullfile(data_path,'*.mat'));
for i=1:length(data_files)
    load(fullfile(data_path,data_files(i).name));
    speed = sudushuju;
    speed(speed<=0)=1;
    speed(speed>100)=100;
    if i<10
        save(fullfile(augment_path, ['000',num2str(i),'.mat']), 'speed')
    end
    if i>=10&&i<100
        save(fullfile(augment_path, ['00',num2str(i),'.mat']) ,'speed')
    end
    if i>=100&&i<1000
        save(fullfile(augment_path, ['0',num2str(i),'.mat']) ,'speed')
    end
    if i>=1000
        save(fullfile(augment_path, [num2str(i),'.mat']) ,'speed')
    end
end
for i=1:length(data_files)
    load(fullfile(data_path,data_files(i).name));
    speed = speed + 20*rand(size(speed)) - 10;
    %speed = sudushuju + 5*rand() - 2.5;
    speed(speed<=0)=5 + rand();
    speed(speed>100)= 100 - 10*rand();
    k=i+252;
    if k<10
        save(fullfile(augment_path, ['000',num2str(k),'.mat']), 'speed')
    end
    if k>=10&&k<100
        save(fullfile(augment_path, ['00',num2str(i),'.mat']) ,'speed')
    end
    if k>=100&&k<1000
        save(fullfile(augment_path, ['0',num2str(k),'.mat']) ,'speed')
    end
    if k>=1000
        save(fullfile(augment_path, [num2str(k),'.mat']) ,'speed')
    end
end

for i=1:length(data_files)
    load(fullfile(data_path,data_files(i).name));
    speed = speed + 20*rand(size(speed)) - 10;
    speed(speed<=0)=5 + rand();
    speed(speed>100)= 100 - 10*rand();
    k=i+504;
    if k<10
        save(fullfile(augment_path, ['000',num2str(k),'.mat']), 'speed')
    end
    if k>=10&&k<100
        save(fullfile(augment_path, ['00',num2str(i),'.mat']) ,'speed')
    end
    if k>=100&&k<1000
        save(fullfile(augment_path, ['0',num2str(k),'.mat']) ,'speed')
    end
    if k>=1000
        save(fullfile(augment_path, [num2str(k),'.mat']) ,'speed')
    end
end