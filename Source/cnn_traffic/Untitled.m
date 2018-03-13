figure()
plot(pred_m(5,:))
hold on
plot(real_m(5,:))
legend('MCNN', 'Ground-truth')

grid on