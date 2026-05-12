clc; clear; close all;

% Load data
df = readtable('prediction_n18.csv');

% Extract confidence column and remove NaNs
data = df.confidence;
data = data(~isnan(data));

% Mean and standard deviation
mu = mean(data);
sigma = std(data);

% Create histogram (density normalization)
figure;
histogram(data, 20, 'Normalization', 'pdf', ...
    'FaceColor', [0 0.4470 0.7410], ...
    'FaceAlpha', 0.5);
hold on;

% Normal distribution curve
x = linspace(min(data), max(data), 100);
y = normpdf(x, mu, sigma);

plot(x, y, 'r', 'LineWidth', 2);

% Labels and title
title("Normal Distribution of Model Confidence");
xlabel("Confidence");
ylabel("Density");
grid on;
legend("Histogram", "Normal Fit");