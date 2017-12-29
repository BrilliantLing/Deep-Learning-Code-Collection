%clear all
%load('D:\MasterDL\trans\yabx\figure\ground_truth_day30\day30_result.mat')

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
plot(pred_m(42,:))
hold on
plot(real_m(42,:))
legend('MSCNN', 'Ground-truth')

grid on