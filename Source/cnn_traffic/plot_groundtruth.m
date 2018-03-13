clear all
load('D:\MasterDL\trans\yabx\new_test_log\pred\66.mat')
load('D:\MasterDL\trans\yabx\new_test_log\real\66.mat')

figure()
%plot(ann(10,:))
%hold on
%plot(arima(10,:))
%hold on
%plot(cnn(10,:))
%hold on
%plot(knn(10,:))
%hold on 

%nhnx 15,19,30,
plot(pred_m(12,:))
hold on
plot(real_m(12,:))
legend('MSCNN', 'Ground-truth')

grid on