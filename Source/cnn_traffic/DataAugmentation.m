data_path = 'D:\Test\2011NHNX_speed_fix';
augment_path = 'D:\Test\2011NHNX_speed_augment';
data_files = dir(fullfile(data_path,'*.mat'));
for i=1:length(data_files)
    load(fullfile(data_path,data_files(i).name));
    speed = sudushuju;
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
    speed = sudushuju + 6*rand(size(sudushuju)) - 3;
    %speed = sudushuju + 5*rand() - 2.5;
    speed(speed<=0)=1;
    k=i+365;
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
    speed = sudushuju + 10*rand(size(sudushuju)) - 5;
    speed(speed<=0)=1;
    k=i+730;
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