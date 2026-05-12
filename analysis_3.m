clc; clear; close all 

% -- Load data --  
data = readtable('prediction_n10.csv'); 

% -- True labels -- 
labels = strcmp(data.real, 'prostate');  % replace (i.e., 'control', 'prostate', 'skin') for the diff plot

% -- Probability predicted --
scores = data.prostate;                  % replace (i.e., data.control, data.prostate, data.skin) for the diff plot
threshold = unique(sort(scores, 'descend'));

% -- Temp variable -- 
TPR = zeros(length(threshold), 1); 
FPR = zeros(length(threshold), 1); 

% -- Threshold sweep --
for i=1:length(threshold)
    t = threshold(i);
    
    pred = scores >= t; 
    
    TP = sum(labels==1 & pred==1); 
    FP = sum(labels==0 & pred==1); 
    TN = sum(labels==0 & pred==0); 
    FN = sum(labels==1 & pred==0); 

    TPR(i) = TP / (TP + FN); 
    FPR(i) = FP / (FP + TN);
    
    
end

hold on
plot(FPR, TPR, '-o', 'Color', 'b', 'LineWidth', 2)
title("ROC - Prostate", "FontSize", 12);
xlabel("FPR", "FontSize", 12); ylabel("TPR", 'FontSize', 12);

