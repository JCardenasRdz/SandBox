function [ OUT ] = cest_map_anetta( CESTdata, ppm, mask )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% find voxels to process
index_voxels = mask;
% turn into matrix
CESTdata_matrix = reshape(CESTdata, [], length(ppm) );


% eliminate stuff we don't need
CESTdata_matrix = CESTdata_matrix(index_voxels,:);

% fit
LB=[0.5,0,0, .3,.3,.3, -1, 4, 8]';
x0=[0.9,.1,.1, 1,1,1, 0, 5, 9.5]';
UB=[1.0,0.5,0.5, 5,5,5, 1, 6, 11]';

Method.Npools =2;
Method.range =[1,length(ppm)];
Method.x0  =[LB,x0,UB];

num_fits = size(CESTdata_matrix,1);

num_fits
OUT = zeros(num_fits, 3);

for j=1:1:num_fits
    z = CESTdata_matrix(j,:)';
    [CESTfit] =  cf_Lorentzian( z,ppm,Method, max(z));
    
    OUT(j,:) = CESTfit.pars(1:3)'; 

end




end

