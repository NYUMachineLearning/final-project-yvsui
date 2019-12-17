%% Machine learning class final project
% Data cleaning and EDA
temp = matlab.desktop.editor.getActive;
cd(fileparts(temp.Filename));
clear all; close all; clc

%% Read data, specify region numbers
[MK_all,data_txt,data_raw] = xlsread('MK_all_91219.xlsx');
[cog_all,cog_header,cog_raw] = xlsread('AnalyzableData_8.1.19.xlsx');

header = data_txt(1,:)';
cog_ID = 13:17;
group_ID   = 1:12;
ROI_subcor = [20:23;32:35];  %left;right
ROI_CC     = fliplr(108:112);
ROI_ctx    = [40:73;74:107]; %left;right
ROI_wm     = [113:146;147:180];  %left;right

subj_grp  = data_raw(2:end,group_ID);


%% Look at distribution, find outliers
figure
options.handle     = figure(1);
options.color_area = [243 169 114]./255;    % Orange theme
options.color_line = [236 112  22]./255;
options.alpha      = 0.5;
options.line_width = 2;
options.error      = 'std';

subplot(1,2,1)
plot_areaerrorbar(MK_all(:,ROI_ctx(1,:))); hold on
plot_areaerrorbar(MK_all(:,ROI_wm(1,:)), options); 
title('Left hemisphere');ylim([-2,3]);axis square;xlabel('left hemisphere regions')
subplot(1,2,2)
plot_areaerrorbar(MK_all(:,ROI_ctx(2,:))); hold on
plot_areaerrorbar(MK_all(:,ROI_wm(2,:)),options); 
legend({'Cortical areas','White matter areas'},'location','southeast')
title('Right hemisphere');ylim([-2,3]);axis square;xlabel('right hemisphere regions')

figure
x=[1,2,2];y=[13,3,11];
for ii=1:3
    subplot(1,4,ii)
    boxplot(MK_all(:,ROI_ctx(x(ii),y(ii))))
    title(header{ROI_ctx(x(ii),y(ii))})
end

subplot(1,4,4)
boxplot(MK_all(:,ROI_wm(1,15)))
title(header{ROI_wm(1,15)})
% found outliers in 1009, 1039, 1062, 1095

%% Outliers!
% %take out observations
% outliers = [1009,1039,1062,1095];
% for ii = 1:length(outliers)
%     if sum(MK_all(:,1)==outliers(ii))
%         tmp = find(MK_all(:,1)==outliers(ii));
%         MK_all(tmp,:) = [];
%         subj_grp(tmp,:) = [];
%         cog_all(find(cog_all(:,1)==outliers(ii)),:) = [];
%     end
% end

% remove outlier regions
ROI_ctx(:,y)=[];
ROI_wm(:,15)=[];

lack_cog = [1003 1009 1035 1039 1055 1060 1062 1080 1095 1210];
for ii = 1:length(lack_cog)
    if sum(MK_all(:,1)==lack_cog(ii))
        tmp = find(MK_all(:,1)==lack_cog(ii));
        MK_all(tmp,:) = [];
        subj_grp(tmp,:) = [];
    end
    if sum(cog_all(:,1)==lack_cog(ii))
        cog_all(find(cog_all(:,1)==lack_cog(ii)),:) = [];
    end
end

%%
figure
options.handle     = figure(1);
options.color_area = [243 169 114]./255;    % Orange theme
options.color_line = [236 112  22]./255;
options.alpha      = 0.5;
options.line_width = 2;
options.error      = 'std';

subplot(1,2,1)
plot_areaerrorbar(MK_all(:,ROI_ctx(1,:))); hold on
plot_areaerrorbar(MK_all(:,ROI_wm(1,:)), options); 
title({'Outlier removed'});ylim([0.4,1.2]);axis square;xlabel('left hemisphere regions')
subplot(1,2,2)
plot_areaerrorbar(MK_all(:,ROI_ctx(2,:))); hold on
plot_areaerrorbar(MK_all(:,ROI_wm(2,:)),options); 
legend({'Cortical areas','White matter areas'},'location','southeast')
ylim([0.4,1.2]);axis square;xlabel('right hemisphere regions')

%%
% a=[a,zeros(107,1)];
% for ii = 1:107
%     if sum(MK_all(:,2)==a(ii,1))
%     tmp = find(MK_all(:,2)==a(ii,1));
%     a(ii,2) = MK_all(tmp,1);
%     end
% end

%% PCA cognitive score
cog_ID = 6:size(cog_all,2);
cog_header = cog_header(1,cog_ID);
for ii=6:8 % 6-9 are raw scores need to get to z scores
    tmp = cog_all(:,ii);
    if sum(isnan(tmp))
        a=find(isnan(tmp));
        tmp(a)=[];
        tmp=zscore(tmp);
        tmp=[tmp(1:a-1);cog_all(a,ii);tmp(a:end)];
    else
        tmp=zscore(tmp);       
    end
    cog_all(:,ii) = tmp;
end

[pca_coef,pc_score,latent,tsquared,explained,cog_mean] = pca(cog_all(:,cog_ID),...
    'algorithm','als');
load('workspace_after_PCA.mat');
figure
subplot(2,1,1)
plot(latent,'-o'); title('Scree plot')
ylabel('Eigenvalue');axis square
subplot(2,1,2)
tmp=zeros(size(explained));
for ii = 1:length(explained)
    tmp(ii)=sum(explained(1:ii));
end
plot(tmp,'-o'); axis square
xlabel('Component number'); ylabel('Explained variance')

figure
h = biplot(pca_coef(:,1:3));
axis square %,'scores',pc_score(:,1:3),'varlabels',cog_header{cog_ID}
for ii = 1:length(cog_ID)
    h(ii).Color = [0,.447,.741]; 
    h(ii+length(cog_ID)).Color = [0,.447,.741]; 
end

%look at loadings - trying to name PCs
loadings = zeros(length(cog_ID),1);
for ii = 1:length(cog_ID)
    loadings(ii,:) = find(abs(pca_coef(ii,1:3))==max(abs(pca_coef(ii,1:3))));
end

%% between-region correlation
lobes=cell(7,2); 
for ii=1:2
    if ii==1
        header_tmp=header(ROI_ctx(1,:));else
        header_tmp=header(ROI_wm(1,:));
    end
    lobes{1,ii}=find(contains(header_tmp,'frontal')+contains(header_tmp,'pars')); %'contains' function outputs logical(0/1)
    lobes{2,ii}=find(contains(header_tmp,'cingulate'));
    lobes{3,ii}=find(contains(header_tmp,'insula'));
    lobes{4,ii}=find(contains(header_tmp,'central'));
    lobes{5,ii}=find(contains(header_tmp,'temporal')+contains(header_tmp,'banks')+contains(header_tmp,'entorh')+contains(header_tmp,'fusiform')+contains(header_tmp,'parahipp'));
    lobes{6,ii}=find(contains(header_tmp,'parietal')+contains(header_tmp,'precuneus')+contains(header_tmp,'supramarg'));
    lobes{7,ii}=find(contains(header_tmp,'occipital')+contains(header_tmp,'lingual')+contains(header_tmp,'perical')+strcmp(header_tmp,'-cuneus'));
end
roi_ctx=cell2mat(lobes(:,1));
roi_wm=cell2mat(lobes(:,2));

figure
tmp = MK_all(:,[ROI_ctx(1,roi_ctx(:)),ROI_ctx(2,roi_ctx(:)),ROI_wm(1,roi_wm(:)),ROI_wm(2,roi_wm(:)),ROI_subcor(:)']);
% tmp = MK_all(:,[ROI_ctx(1,:),ROI_ctx(2,:),ROI_wm(1,:),ROI_wm(2,:),ROI_subcor(:)']);
M = corr(tmp);
subplot(1,2,1)
M(find(M==1))=NaN;
imshow(M,[min(M(:)),max(M(:))]);colormap(gca,'parula');axis square;title('MK in 140 regions')
subplot(1,2,2)
tmp = 0.5*(MK_all(:,[ROI_ctx(1,roi_ctx(:)),ROI_wm(1,roi_wm(:)),ROI_subcor(1,:)])+MK_all(:,[ROI_ctx(2,roi_ctx(:)),ROI_wm(2,roi_wm(:)),ROI_subcor(2,:)]));
M = corr(tmp);
M(find(M==1))=NaN;
imshow(M,[min(M(:)),max(M(:))]);colormap(gca,'parula');axis square;title('MK in 70 regions (L&R average)')





