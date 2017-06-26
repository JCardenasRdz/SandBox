clc; close all; clear all;
% define directories and variables
cest_dir = 'data/AVG-uPA-T1-AL1.HY1/3';

% load data
[CEST, info] = Bruker_reader(cest_dir);

% turn it into 3D
CEST = squeeze(CEST);

% xdata
ppm = info.cest_array / 300;
% Z-dimension
[~,~, num_images] = size(CEST);
%% select ROI and get mask
imshow(CEST(:,:,1),[]); mask = roipoly;
close all
%% get average signals
zspectra_vector = avgroi(CEST, mask);
zspectra_matrix = reshape(zspectra_vector,length(ppm),[]);
%% fit lorentzian pools, 9.5, 5.0, 0.0
%%
LB=[0.5,0,0, .3,.3,.3, -1, 4, 8]';
x0=[0.9,.1,.1, 1,1,1, 0, 5, 9.5]';
UB=[1.0,0.5,0.5, 5,5,5, 1, 6, 11]';

Method.Npools =2;
Method.range =[1,length(ppm)];
Method.x0  =[LB,x0,UB];
%%
%mean
Z = mean(zspectra_matrix(:,7:end),2);
% normalize
[CESTfit] =  cf_Lorentzian( (Z),ppm,Method, max(Z));      
x = CESTfit.ppmadj';
y1 = CESTfit.Zspectrum *100;
y2 = CESTfit.Lsum * 100;
y3 = CESTfit.cfitall(:,2) * 100;
y4 = CESTfit.cfitall(:,3) * 100;
% write 
% col 01 = ppm
% col 02 = raw data ( blue dots)
% col 03 = sum of lorentzians (fitted, blue line)
% col 04 = 5 ppm (fitted, red line)
% col 05 = 9.5 ppm (fitted, red line)
%%
subplot(1,2,1);
plot(x,y1,x,y2)
subplot(1,2,2);
plot(x,y3,x,y4)
%%
[ OUT ] = cest_map_anetta( CEST(:,:,507:end), ppm, mask );
%%
A2 = zeros(96,96);
A2(mask) = OUT(:,2);
imshow(A2)
colormap('jet')
