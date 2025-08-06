% Prediction of fault type and severity based on given features

% Defining the file
datafolder = "C:\Users\btn9n\Downloads\CWUR_data";
fileList = dir(fullfile(datafolder, '*.mat'));

% File listing
for i = 1:length(fileList)
    disp(fileList(i).name)
end

% Initializing storage
allData = [];

for i = 1:length(fileList)
    filename = fileList(i).name;
    filepath = fullfile(datafolder, filename);
    
    % Load variables from the file
    vars = whos('-file', filepath);

    % Find the DE signal variable
    varName = '';
    for v = 1:length(vars)
        if contains(vars(v).name, 'DE')
            varName = vars(v).name;
            break;
        end
    end

    % Load and extract the signal
    load(filepath, varName);
    signal = eval(varName);

    fs = 12000;  % Sampling frequency
    signal = signal(1:fs);  % Trim to first 1 second

    % Time-domain features
    RMS = rms(signal);
    SK = skewness(signal);
    KUR = kurtosis(signal);
    MN = mean(signal);
    STD = std(signal);

    % Perform FFT
    L = length(signal); % total number of samples
    Y = abs(fft(signal));
    f = fs*(0:(L/2))/L;       % one-sided frequency array
    Y = Y(1:length(f));       % take only one-sided FFT
    Y_norm = Y / sum(Y);      % normalize
  
    
    % Now compute scalar frequency-domain features
    [~, peakIdx] = max(Y);
    PeakFreq = f(peakIdx);    % single value
    
    SpectralCentroid = sum(f .* Y_norm);                                
    SpectralSpread = sqrt(sum(((f - SpectralCentroid).^2) .* Y_norm));  
    SpectralFlatness = geomean(Y + eps) / mean(Y + eps);                

    if numel(SpectralCentroid) > 1
        SpectralCentroid = mean(SpectralCentroid);
    end
    if numel(SpectralSpread) > 1
        SpectralSpread = mean(SpectralSpread);
    end
    if numel(SpectralFlatness) > 1
        SpectralFlatness = mean(SpectralFlatness);
    end
    % Label fault type
    if contains(filename, 'IR')
        fault = "InnerRace";
    elseif contains(filename, 'B')
        fault = "Ball";
    elseif contains(filename, 'OR')
        fault = "OuterRace";
    else
        fault = "Normal";
    end

    % Label fault severity
    if contains(filename, '007')
        severity = "Small";
    elseif contains(filename, '014')
        severity = "Medium";
    elseif contains(filename, '021')
        severity = "Large";
    else
        severity = "Healthy";
    end

    % Store row in table
    tempTable = table(RMS, SK, KUR, MN, STD, PeakFreq, ...
        SpectralCentroid, SpectralSpread, SpectralFlatness, ...
        string(fault), string(severity), string(filename), ...
        'VariableNames', {'RMS', 'Skewness', 'Kurtosis', 'Mean', 'StdDev', ...
        'PeakFreq', 'SpectralCentroid', 'SpectralSpread', 'SpectralFlatness', ...
        'FaultType', 'Severity', 'FileName'});

    allData = [allData; tempTable];
end
% Preview result
disp(allData(1:5, :))

% Save features to CSV
writetable(allData, 'bearing_total_features.csv');

% Read back the data
data = readtable('bearing_total_features.csv');

% Plot for the last signal
t = (0:length(signal)-1) / fs; %create a time array using sampling freq
figure;
subplot(2,1,1);
plot(t, signal); %plot raw signal over time for 1 sec
xlabel('Time (s)');
ylabel('Amplitude');
title(['Time Domain: ' filename]);
grid on;

subplot(2,1,2);
plot(f, Y); %energy distribution across frequencies
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title(['Frequency Domain: ' filename]);
grid on;

% Extract features
features = data{:, {'RMS','Skewness','Kurtosis','Mean','StdDev', ...
    'PeakFreq','SpectralSpread','SpectralFlatness'}};

%since spectralcentroid is coming out to be a constant we are dropping it
%in further analysis

% Data visualization (EDA)
% Correlation heatmap between selected features
selectedFeatures = {'RMS','Skewness','Kurtosis','Mean','StdDev','PeakFreq','SpectralSpread','SpectralFlatness'};
corrMatrix = corr(data{:, selectedFeatures});

figure;
heatmap(selectedFeatures, selectedFeatures, ...
    corrMatrix, 'Colormap', parula, 'ColorLimits', [-1, 1]);
title('Feature Correlation Heatmap');

%removing StdDev and Spectralflatness as they have high correlation with
%rms
%-------------------------------------------------------------------------------------------------------------------------------------------
%% 

fprintf('Model to predict fault type')

%pairwise scatterplot grouped by faults
labels_fault=data.FaultType;
featureNames={'RMS','Skewness','Kurtosis','Mean','PeakFreq','SpectralSpread'};
features_new=data{:,featureNames};
figure;
[~, ax]=gplotmatrix(features_new, [], labels_fault,[],[],[], false);
sgtitle('Feature Relationships by Fault Type');
for i=1:length(featureNames)
    ylabel(ax(i,1), featureNames{i},'FontWeight','bold');
    xlabel(ax(end,i), featureNames{i},'FontWeight','bold');
end

%boxplot for each feature. shows how distribution of each feature varies by fault type

figure;
subplot(2,3,1); boxplot(data.RMS, data.FaultType); title('RMS');
subplot(2,3,2); boxplot(data.Skewness, data.FaultType); title('Skewness');
subplot(2,3,3); boxplot(data.Kurtosis, data.FaultType); title('Kurtosis');
subplot(2,3,4); boxplot(data.Mean, data.FaultType); title('Mean');
subplot(2,3,5); boxplot(data.PeakFreq, data.FaultType); title('PeakFreq');
subplot(2,3,6); boxplot(data.SpectralSpread, data.FaultType); title('SpectralSpread');
sgtitle('Feature Distributions by Fault Type');


%% 

%training the model

%labels_fault= data.FaultType;
labels_fault=categorical(labels_fault);

rng(42); %fixing a random seed
%split of train and test data
cv=cvpartition(labels_fault, 'HoldOut', 0.2);
x_train= features_new(training(cv),:);
x_test= features_new(test(cv), :);
y_train=labels_fault(training(cv),:);
y_test=labels_fault(test(cv),:);

%decision tree
faulttreeModel=fitctree(x_train, y_train,'CrossVal','on','KFold',5);
accTree= 1- kfoldLoss(faulttreeModel);
disp(['Decision Tree CV Accuracy:', num2str(accTree*100, '%.2f'), '%']);

%k-NN
faultKNNmodel=fitcknn(x_train, y_train, 'NumNeighbors', 5, 'CrossVal','on','KFold',5);
accKNN=1-kfoldLoss(faultKNNmodel);
disp(['k-NN CV Accuracy:', num2str(accKNN*100, '%.2f'), '%']);

%SVM
template = templateSVM('KernelFunction','linear');
faultSVMmodel=fitcecoc(x_train,y_train,'Learners',template,'CrossVal','on','KFold',5);
accSVM=1-kfoldLoss(faultSVMmodel);
disp(['SVM CV Accuracy:', num2str(accSVM*100, '%.2f'), '%']);

%Bagged Trees
template=templateTree();
hyperOpts = struct('Optimizer','bayesopt', ...
                   'AcquisitionFunctionName','expected-improvement-plus', ...
                   'ShowPlots', true, ...
                   'Verbose', 1, ...
                   'KFold', 5);
faultBaggedmodel = fitcensemble(x_train, y_train, ...
    'Method', 'Bag', ...
    'Learners', template, ...
    'OptimizeHyperparameters', {'NumLearningCycles', 'MinLeafSize', 'MaxNumSplits'}, ...
    'HyperparameterOptimizationOptions', hyperOpts);
accBagged = (1 - faultBaggedmodel.HyperparameterOptimizationResults.MinObjective);
disp(['Bagged Trees CV Accuracy:', num2str(accBagged*100, '%.2f'), '%']);

%bagged trees have max cv accuracy. hence choosing it to train the model

%training final model on complete train dataset
Final_fault_model=fitcensemble(x_train, y_train, 'Method','Bag',...
    'NumLearningCycles',100,'Learners',template);
%% 

%prediction on test set
[y_pred, scores]=predict(Final_fault_model, x_test);

%calculating accuracy
testAccuracy=mean(y_pred==y_test)*100;
fprintf('Test Accuracy (Bagged Trees for fault type): %.2f%%\n', testAccuracy);

%understanding the model performance

%confusion matrix
confusionchart(y_test, y_pred);
title('Confusion Matrix for Fault Type prediction-Bagged Trees');

%classification report
c=confusionmat(y_test,y_pred);
precision=diag(c)./sum(c,2);
recall=diag(c)./sum(c,1)';
f1=2*(precision.*recall)./(precision+recall);
disp(table(precision, recall, f1, 'VariableNames',{'Precision','Recall','F1'}));

% True labels
trueLabels = y_test; % categorical

classes = Final_fault_model.ClassNames; % class labels, e.g., ["InnerRace","Ball","OuterRace","Normal"]
nClasses = numel(classes);

for i = 1:nClasses
    % One-vs-rest binary labels for class i
    trueBinary = (trueLabels == classes(i));
    
    % Scores/probabilities for class i
    scoreClass = scores(:, i);
    
    % Compute ROC curve and AUC
    [X, Y, ~, AUC] = perfcurve(trueBinary, scoreClass, true);
    
    % Plot ROC curve for this class
    figure;
    plot(X, Y, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC Curve for class ', char(classes(i)), ' (AUC = ', num2str(AUC, '%.2f'), ')']);
    grid on;
end
%feature importance
importance=predictorImportance(Final_fault_model);
figure;
bar(importance);
xlabel('Feature Index');
ylabel('Importance');
title('Feature Importance (Fault Type)- Bagged Trees')
save('Bagged_fault_type_classifier.mat','Final_fault_model');

%--------------------------------------------------------------------------------------------------------------------------------------------
%% 

fprintf('Model to predict severity')

 
%pairwise scatterplot grouped by severity

labels_severity=data.Severity;

figure;
[~, ax]=gplotmatrix(features_new, [], labels_severity,[],[],[], false);
sgtitle('Feature Relationships by Severity');
for i=1:length(featureNames)
    ylabel(ax(i,1), featureNames{i},'FontWeight','bold');
    xlabel(ax(end,i), featureNames{i},'FontWeight','bold');
end

%boxplot for each feature by severity
figure;
subplot(2,3,1); boxplot(data.RMS, data.Severity); title('RMS');
subplot(2,3,2); boxplot(data.Skewness, data.Severity); title('Skewness');
subplot(2,3,3); boxplot(data.Kurtosis, data.Severity); title('Kurtosis');
subplot(2,3,4); boxplot(data.Mean, data.Severity); title('Mean');
subplot(2,3,5); boxplot(data.PeakFreq, data.Severity); title('PeakFreq');
subplot(2,3,6); boxplot(data.SpectralSpread, data.Severity); title('SpectralSpread');
sgtitle('Feature Distributions by Severity');


%% 

%training the model

labels_severity=categorical(labels_severity);

rng(42); %fixing a random seed
%split of train and test data
cv_sev=cvpartition(labels_severity, 'HoldOut', 0.2);
x_train_sev= features_new(training(cv_sev),:);
x_test_sev= features_new(test(cv_sev), :);
y_train_sev=labels_severity(training(cv_sev),:);
y_test_sev=labels_severity(test(cv_sev),:);

%decision tree
severitytreeModel=fitctree(x_train_sev, y_train_sev,'CrossVal','on','KFold',5);
accTree= 1- kfoldLoss(severitytreeModel);
disp(['Decision Tree CV Accuracy:', num2str(accTree*100, '%.2f'), '%']);

%k-NN
severityKNNmodel=fitcknn(x_train_sev, y_train_sev, 'NumNeighbors', 5, 'CrossVal','on','KFold',5);
accKNN=1-kfoldLoss(severityKNNmodel);
disp(['k-NN CV Accuracy:', num2str(accKNN*100, '%.2f'), '%']);

%SVM
template = templateSVM('KernelFunction','linear');
severitySVMmodel=fitcecoc(x_train_sev,y_train_sev,'Learners',template,'CrossVal','on','KFold',5);
accSVM=1-kfoldLoss(severitySVMmodel);
disp(['SVM CV Accuracy:', num2str(accSVM*100, '%.2f'), '%']);

%Bagged Trees
template=templateTree('MaxNumSplits',20);
severityBaggedmodel=fitcensemble(x_train_sev, y_train_sev, 'Method','Bag',...
    'NumLearningCycles',100,'Learners',template,'CrossVal','on','KFold',5);
accBagged=1-kfoldLoss(severityBaggedmodel);
disp(['Bagged Trees CV Accuracy:', num2str(accBagged*100, '%.2f'), '%']);

%bagged trees have max cv accuracy. hence choosing it to train the model

%training final model on complete train dataset
Final_severity_model=fitcensemble(x_train_sev, y_train_sev, 'Method','Bag',...
    'NumLearningCycles',100,'Learners',template);
%% 

%prediction on test set
[y_pred_sev,scores_sev]=predict(Final_severity_model, x_test_sev);

%calculating accuracy
testAccuracy=mean(y_pred_sev==y_test_sev)*100;
fprintf('Test Accuracy (Bagged Trees for severity): %.2f%%\n', testAccuracy);

%understanding the model performance

%confusion matrix
confusionchart(y_test_sev, y_pred_sev);
title('Confusion Matrix for Severity prediction-Bagged Trees');

%classification report
c=confusionmat(y_test_sev,y_pred_sev);
precision=diag(c)./sum(c,2);
recall=diag(c)./sum(c,1)';
f1=2*(precision.*recall)./(precision+recall);
disp(table(precision, recall, f1, 'VariableNames',{'Precision','Recall','F1'}));

%AUC_ROC
trueLabels_sev = y_test_sev; % categorical
classes_sev = Final_severity_model.ClassNames;
nClasses_sev = numel(classes_sev);

for i = 1:nClasses_sev
    trueBinary = (trueLabels_sev == classes_sev(i));
    scoreClass = scores_sev(:, i);
    [X, Y, ~, AUC] = perfcurve(trueBinary, scoreClass, true);
    figure;
    plot(X, Y, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC Curve for Severity class ', char(classes_sev(i)), ' (AUC = ', num2str(AUC, '%.2f'), ')']);
    grid on;
end

%feature importance
importance=predictorImportance(Final_severity_model);
figure;
bar(importance);
xlabel('Feature Index');
ylabel('Importance');
title('Feature Importance (Severity)- Bagged Trees')

%saving the model for further use
save('Bagged_severity_classifier.mat','Final_severity_model');

%------------------------------------------------------------------------------------------------------------------------------------------------------


%% 

% Example feature row from your dataset:
sample_features = [0.14, 0.017, 2.6836, 0.0062, 3578, 1.82e-12];

sample_features = reshape(sample_features, 1, []);  % Ensure 1x8 format

[predType, predSeverity] = predictBearingFaultFromFeatures(sample_features, Final_fault_model, Final_severity_model);

disp(['Predicted Fault Type: ', char(predType)]);
disp(['Predicted Fault Severity: ', char(predSeverity)])

function [predictedType, predictedSeverity] = predictBearingFaultFromFeatures(features, faultModel, severityModel)


    % Prediction
    predictedType = predict(faultModel, features);
    predictedSeverity = predict(severityModel, features);
end
