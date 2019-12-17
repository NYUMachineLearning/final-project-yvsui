%% Machine learning class final project
% Support vector machine final project
% One main challenge of this binary classification project is the 
temp = matlab.desktop.editor.getActive;
cd(fileparts(temp.Filename));
clear all; close all; clc

%% SVM with features weighted as important using feature selection algorithm
% load data
[all_mk_train,all_header_train,~] = xlsread('MK_data.xlsx','all_train');
[all_mk_test,all_header_test,~] = xlsread('MK_data.xlsx','all_test');
all_label_train = all_header_train(2:end,2:3);
all_label_test = all_header_test(2:end,2:3);
all_header = all_header_train(1,4:end);
clear all_header_*

all_mk_train(:,1:3)=[]; %take out the columns that are labels and subject ID
all_mk_test(:,1:3)=[];

all_resp_train = strcmp(all_label_train(:,1),'C');
all_resp_test = strcmp(all_label_test(:,1),'C');

% Feature selection using neighborhood components analysis
importance = fscnca(all_mk_train,all_resp_train);
figure
subplot 121; plot(importance.FeatureWeights,'ro');
xlabel('Feature index'); grid on; ylabel('Feature weight')
subplot 122; plot(log(importance.FeatureWeights),'ro');
xlabel('Feature index'); grid on; ylabel('log Feature weight')
% chose -14 as threshold for feature selection (log scale feature weight)
feature_ID = find(log(importance.FeatureWeights)>-14);
fprintf('Features selected:\n')
disp(all_header(feature_ID)')

all_mk_train = all_mk_train(:,feature_ID); 
all_mk_test = all_mk_test(:,feature_ID); 

rng(3) %seeds random number generator

% setting cross-validation parameters (5-fold cv)
c = cvpartition(size(all_mk_train,1),'KFold',5);

% train SVM ------------------------------------------------------------
% Bayes optimization
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% Mdl = fitcsvm(all_mk_train,all_resp_train,'KernelFunction','polynomial',...
%     'OptimizeHyperparameters',{'PolynomialOrder','BoxConstraint'},'HyperparameterOptimizationOptions',opts)

% no Bayes
Mdl_all = fitcsvm(all_mk_train,all_resp_train,'KernelFunction','polynomial',...
    'OptimizeHyperparameters',{'PolynomialOrder','BoxConstraint'},...
    'HyperparameterOptimizationOptions',struct('CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus'))

% test -----------------------------------------------------------------
[label_pred,score_pred] = predict(Mdl_all,all_mk_test); 
% [label_pred,score_pred] = predict(Mdl_at_chance,all_mk_test); 
[X,Y,~,AUC] = perfcurve(all_resp_test,score_pred(:,2),'true');

% testing ROC
figure;plot(X,Y,'linewidth',2);axis square; hold on
xlabel('False positive rate'); ylabel('True positive rate')
title('Testing ROC'); xlim([-0.02,1.02]); ylim([-0.02,1.02]); 
plot([0,1],[0,1],'--','color',[.8,.8,.8]); hold on
text(0.6,0.1,sprintf('AUC=%.3f',AUC)); hold off

% confusion matrix
prediction = repmat({'S'},length(label_pred),1);
prediction(label_pred)={'C'};
prediction = [string(all_label_test),prediction];
figure;plotconfusion(all_resp_test',label_pred')
set(gca,'xticklabel',{'Patient','Control','',''},'yticklabel',{'Patient','Control',''})

% final model (which I used for project report and presentation)
% |===============================================================================================|
% |Iter| Eval   | Objective  | Objective  | BestSoFar  | BestSoFar  | BoxConstrain-| PolynomialOr-|
% |    | result |            | runtime    | (observed) | (estim.)   | t            | der          |
% |===============================================================================================|
% | 27 | Accept |    0.31481 |    0.05644 |    0.31481 |    0.31483 |       7.5321 |            2 |


%% SVM using features suggested in schizophrenia literature
% load data
[mk_train,header_train,~] = xlsread('MK_data.xlsx','selected_train');
[mk_test,header_test,~] = xlsread('MK_data.xlsx','selected_test');
label_train = header_train(2:end,2:3);
label_test = header_test(2:end,2:3);
header = header_train(1,4:end);
clear header_*

mk_train(:,1:3)=[]; %take out the columns that are labels and subject ID
mk_test(:,1:3)=[];

resp_train = strcmp(label_train(:,1),'C');
resp_test = strcmp(label_test(:,1),'C');

fprintf('Features selected:\n')
disp(header')

rng(13)

% setting cross-validation parameters (5-fold cv)
c = cvpartition(size(mk_train,1),'KFold',5);

% train SVM ------------------------------------------------------------
% Bayes optimization
% opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% Mdl = fitcsvm(mk_train,resp_train,'KernelFunction','polynomial',...
%     'OptimizeHyperparameters',{'PolynomialOrder','BoxConstraint'},'HyperparameterOptimizationOptions',opts)

% no Bayes
Mdl_selected = fitcsvm(mk_train,resp_train,'KernelFunction','polynomial',...
    'OptimizeHyperparameters',{'PolynomialOrder','BoxConstraint'},...
    'HyperparameterOptimizationOptions',struct('CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus'))

% test -----------------------------------------------------------------
[label_pred,score_pred] = predict(Mdl_selected,mk_test); 
[X,Y,~,AUC] = perfcurve(resp_test,score_pred(:,2),'true');

% testing ROC and confusion matrix
figure;plot(X,Y,'linewidth',2);axis square; hold on
xlabel('False positive rate'); ylabel('True positive rate')
title('Testing ROC'); xlim([-0.02,1.02]); ylim([-0.02,1.02]); 
plot([0,1],[0,1],'--','color',[.8,.8,.8]); hold on
text(0.6,0.1,sprintf('AUC=%.3f',AUC)); hold off

prediction = repmat({'S'},length(label_pred),1);
prediction(label_pred)={'C'};
prediction = [string(label_test),prediction];
figure;plotconfusion(resp_test',label_pred')
set(gca,'xticklabel',{'Patient','Control','',''},'yticklabel',{'Patient','Control',''})

% final model (which I used for project report and presentation)
% |===============================================================================================|
% |Iter| Eval   | Objective  | Objective  | BestSoFar  | BestSoFar  | BoxConstrain-| PolynomialOr-|
% |    | result |            | runtime    | (observed) | (estim.)   | t            | der          |
% |===============================================================================================|
% | 22 | Accept |    0.24074 |   0.062436 |    0.24074 |    0.24028 |      0.40716 |            2 |














