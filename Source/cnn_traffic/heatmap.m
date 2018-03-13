clear all
figure
real_data_path = 'D:\MasterDL\trans\nbdx\new_test_log\real\';
pred_data_path = 'D:\MasterDL\trans\nbdx\new_test_log\pred\';
real_data_files = dir(fullfile(real_data_path,'*.mat'));
pred_data_files = dir(fullfile(pred_data_path,'*.mat'));
fig_dir = 'D:\MasterDL\trans\nbdx\new_test_log\all_fig';
for i=1:length(real_data_files)
    load(fullfile(real_data_path,real_data_files(i).name));
    colormap('hot');
    imagesc(real_m);
    saveas(gcf,fullfile(fig_dir, [real_data_files(i).name,'r.jpg']));
    clf
end
for i=1:length(pred_data_files)
    load(fullfile(pred_data_path,pred_data_files(i).name));
    colormap('hot');
    imagesc(pred_m);
    saveas(gcf,fullfile(fig_dir, [pred_data_files(i).name,'p.jpg']));
    clf
end
