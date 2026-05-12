clc; clear; close all

% -- Load data --
data = readtable('prediction_n18.csv'); 
classes = ["breast", "control", "prostate", "skin"]; 

% -- Temp variable -- 
mean_conf = zeros(size(classes)); 

for i=1:length(classes)
    
    k = classes(i); 

    idx = strcmp(data.real, k);

    mean_conf(i) = mean(data{idx, k});

end 

bar(categorical(classes), mean_conf)
ylabel('Mean True-Class Probability')
xlabel('True Class')
ylim([0 1])