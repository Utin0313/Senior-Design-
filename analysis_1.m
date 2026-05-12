clc; clear; close all

% Plot per class 
data = readtable('prediction_n18.csv');
classes = unique(data.pred); 

figure
hold on 
for i = 1:length(classes)
    idx = strcmp(data.pred, classes{i}); 
    cdfplot(data.confidence(idx))
    labels{i} = classes{i};  
end

legend(labels)
xlabel('Confidence')
ylabel('CDF')
title('CDF of confidence per predicted class')
grid on 


% Correct vs incorrect per class 
% Select ONE class
cls = 'skin';

% Filter that class
idx = strcmp(data.pred, cls);
subset = data(idx, :);

% Correct vs incorrect
correct = strcmp(subset.real, subset.pred);

% X axis = sample index
x = 1:height(subset);

figure 
hold on

% Plot correct 
plot(x(correct), subset.confidence(correct), 'o', 'MarkerSize', 8, 'LineWidth', 1.5)
% Plot incorrect 
plot(x(~correct), subset.confidence(~correct), 'x', 'MarkerSize', 8, 'LineWidth', 1.5)

yline(0.7, '--k', 'Threshold = 0.7') 
legend('Correct', 'Incorrect', 'Threshold')
xlabel('Sample Index')
ylabel('Confidence')
title(['Correct vs Incorrect for class: ', cls])
grid on

