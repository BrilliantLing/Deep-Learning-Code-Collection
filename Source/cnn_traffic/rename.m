today_path = 'D:\MasterDL\data_set\traffic_data\2011_yabx_speed\test\tomorrow';
today_mat=dir(fullfile(today_path,'*.mat'));
for i=1:length(today_mat)
    load(fullfile(today_path,today_mat(i).name));
    filename=fullfile(today_path,today_mat(i).name);
    k 1;%= int32(str2num(filename));
    speed = sudushuju;
    if k<10
        save(fullfile(today_path, ['000',num2str(k),'.mat']), 'speed')
    end
    if k>=10&&k<100
        save(fullfile(today_path, ['00',num2str(k),'.mat']) ,'speed')
    end
    if k>=100&&k<1000
        save(fullfile(today_path, ['0',num2str(k),'.mat']) ,'speed')
    end
    if k>=1000
        save(fullfile(today_path, [num2str(k),'.mat']) ,'speed')
    end
end