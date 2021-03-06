clear all

pred_delta = load('tracker_prediction_vel.csv');
real = load('tracker_real.csv');
pred = [real(1,:); real(1:end-1,:) + pred_delta(1:end-1,:)];
const_vel_pred = [real(1,:); real(2,:); real(2:end-1,:) + (real(2:end-1,:)-real(1:end-2,:))];

t=0.0088:0.0088:0.0088*size(pred,1);

figure();
for tracker=1:5
    for axis=1:3
        subplot(5,3,(tracker-1)*3 + axis);
        plot(t,real(:,(tracker-1)*3 + axis));
        hold on
        plot(t,pred(:,(tracker-1)*3 + axis));
        plot(t,const_vel_pred(:,(tracker-1)*3 + axis));
    end
end

resi = real-pred;
err = abs(resi);

csvwrite('TestingResi.csv',resi);

mean(err)
%%
pred_op_delta = load('tracker_operating_prediction_vel.csv');
real_op = load('tracker_operating_real.csv');
pred_op = [real_op(1,:); real_op(1:end-1,:) + pred_op_delta(1:end-1,:)];

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

%%
pred_ocsvm_training_data_delta = load('tracker_ocsvm_training_data_prediction_vel.csv');
tmp_data = load('../data/TrainingDataOCSVM.csv');
real_ocsvm_train = tmp_data(1:size(pred_ocsvm_training_data_delta,1),151:165);
pred_ocsvm_train = [real_ocsvm_train(1,:); real_ocsvm_train(1:end-1,:) + pred_ocsvm_training_data_delta(1:end-1,:)];


resi_ocsvm = real_ocsvm_train-pred_ocsvm_train;

csvwrite('../OneClassSVM/data/ResidualData.csv',resi);