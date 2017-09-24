source_path = 'D:\Test\delta\source';
target_path = 'D:\Test\delta\target';

source_files = dir(fullfile(source_path,'*.mat'));
for i=1:length(source_files)
    load(fullfile(source_path,source_files(i).name));
    prev = speed;
    if i+1<length(source_files)
        load(fullfile(source_path,source_files(i+1).name));
        now = speed;
        delta = now - prev;
        if i<10
            img = mat2gray(delta);
            imwrite(img,fullfile(target_path,['0',num2str(i),'.jpg']));
        end
        if i>=10&&i<100
            img = mat2gray(delta);
            imwrite(img,fullfile(target_path,[num2str(i),'.jpg']));
        end
    else
        delta = 0;
    end
end
