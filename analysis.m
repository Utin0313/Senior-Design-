clc; clear; close all 

% Load data 
data = readtable('prediction_n18.csv');

% Sort data 
avg_data = groupsummary(data, 'real', 'mean', 'confidence'); 
avg_data = sortrows(avg_data, 'mean_confidence', 'descend');

% Plot 
x_axis = categorical(avg_data.real, avg_data.real);
y_axis = avg_data.mean_confidence; 
figure 
bar(x_axis, y_axis)

% Add titles and labels to make it look nice
title('Average Confidence Level by Disease Class');
xlabel('Disease Class');
ylabel('Average Confidence');
ylim([0 1.1]); % Sets the Y-axis limit so bars don't touch the very top

% Add the exact numbers on top of the bars
for i = 1:length(y_axis)
    text(i, y_axis(i) + 0.02, num2str(y_axis(i), '%.3f'), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end


true_classes = [repmat({'Breast'}, 10, 1); ...
                repmat({'Control'}, 10, 1); ...
                repmat({'Prostate'}, 10, 1); ...
                repmat({'Skin'}, 10, 1)];

% 2. Create the Predicted Labels (What the model GUESSED)
% Remember: 5 Breast were guessed as Breast, 5 Breast were guessed as Skin.
% Everything else was guessed perfectly.
predicted_classes = [repmat({'Breast'}, 5, 1); repmat({'Skin'}, 5, 1); ... % The 10 Breast images
                     repmat({'Control'}, 10, 1); ...                       % The 10 Control images
                     repmat({'Prostate'}, 10, 1); ...                      % The 10 Prostate images
                     repmat({'Skin'}, 10, 1)];                             % The 10 Skin images

% 3. Generate the Confusion Matrix Graph
figure;
cm = confusionchart(true_classes, predicted_classes);

% 4. Make it look nice for your presentation
cm.Title = 'Confusion Matrix with n = 10 distinct per class strip';
cm.XLabel = 'Predicted Class';
cm.YLabel = 'True Class';

