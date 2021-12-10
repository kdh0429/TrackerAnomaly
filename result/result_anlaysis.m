clear all

pred = load('tracker_prediction.csv');
real = load('tracker_real.csv');

t=0.0088:0.0088:0.0088*size(pred,1);

figure();
for tracker=1:5
    for axis=1:3
        subplot(5,3,(tracker-1)*3 + axis);
        plot(t,real(:,(tracker-1)*3 + axis));
        hold on
        plot(t,pred(:,(tracker-1)*3 + axis));
    end
end

resi = real-pred;
err = abs(resi);

csvwrite('TestingResi.csv',resi);

mean(err)
%%
pred_op = load('tracker_operating_prediction.csv');
real_op = load('tracker_operating_real.csv');

t_op=0.0088:0.0088:0.0088*size(pred_op,1);

figure();
for tracker=1:5
    for axis=1:3
        subplot(5,3,(tracker-1)*3 + axis);
        plot(t_op,real_op(:,(tracker-1)*3 + axis));
        hold on
        plot(t_op,pred_op(:,(tracker-1)*3 + axis));
    end
end

resi_op = real_op-pred_op;
err_op = abs(resi_op);

mean(err_op)