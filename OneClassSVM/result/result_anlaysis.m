clear all
%%
train = load('training_result.txt');
test = load('testing_result.txt');

figure();

plot(train)
hold on
plot(test)
%%
figure();
response = load('response.txt');
max_response = max(response(:,2));
min_test = min(test)*ones(size(response,1));
threshold_50 = max_response - 1.5*(max_response - min_test);

plot(response(:,1),response(:,2));
hold on
plot(response(:,1),min_test)
plot(response(:,1),threshold_50)

for i=1:size(response,1)
    if response(i,2) < threshold_50
        disp(response(i,2))
        break;
    end
end
