clear all
figure
real_data_path = 'D:\MasterDL\trans\yabx\test_log\reality\';
pred_data_path = 'D:\MasterDL\trans\yabx\test_log\prediction\';
fix_data_path = 'D:\MasterDL\trans\yabx\test_log\predfix\';
delat_data_path = 'D:\MasterDL\trans\yabx\test_log\frdelta';
real_data_files = dir(fullfile(real_data_path,'*.mat'));
pred_data_files = dir(fullfile(pred_data_path,'*.mat'));
fix_data_files = dir(fullfile(fix_data_path,'*.mat'));
delta_data_files = dir(fullfile(delat_data_path,'*.mat'));
fig_dir = 'D:\MasterDL\trans\yabx\test_figures';
deltafig_dir = 'D:\MasterDL\trans\yabx\test_figures\frdelta';
% for i=1:length(real_data_files)
%     load(fullfile(real_data_path,real_data_files(i).name));
%     %figure
%     colormap('hot');
%     imagesc(real_m);
%     saveas(gcf,fullfile(fig_dir, [real_data_files(i).name,'r.jpg']));
%     clf
% end
% for i=1:length(pred_data_files)
%     load(fullfile(pred_data_path,pred_data_files(i).name));
%     %figure
%     colormap('hot');
%     imagesc(pred_m);
%     saveas(gcf,fullfile(fig_dir, [pred_data_files(i).name,'.jpg']));
%     clf
% end
% for i=1:length(fix_data_files)
%     load(fullfile(fix_data_path,fix_data_files(i).name));
%     %figure
%     colormap('hot');
%     imagesc(fix_m);
%     saveas(gcf,fullfile(fig_dir, [fix_data_files(i).name,'f.jpg']));
%     clf
% end
for i=1:length(delta_data_files)
    load(fullfile(delat_data_path,delta_data_files(i).name));
    %figure
    colormap('hot');
    imagesc(fr_delta);
    saveas(gcf,fullfile(deltafig_dir, [delta_data_files(i).name,'f.jpg']));
    clf
end
