clear; close all; clc

config = "config.json";
cData = fileread(config);
cData = jsondecode(cData);
fpath = cData.dataPath;
dataFolder = dir(fpath);
fprintf("Select Folder:\n")
for i = 1:length(dataFolder)
    fprintf("(%d) %s\n",i,dataFolder(i).name)
end
folderSelect = input("Choose Folder number: ");
selectedFolder = fullfile(fpath, dataFolder(folderSelect).name);
lidarFiles = dir(fullfile(selectedFolder, 'lidar_*.pcd'));
fileList = {lidarFiles.name}';
fprintf("Using: \n")
fprintf("\t%s\n",fileList{:});



%% Loop
for i = 1:length(fileList)
    ptcfpath = fullfile(selectedFolder,fileList{i});
    ptCloud = pcread(ptcfpath);

    maxDistance = 0.05;
    [model, inlierIndices, outlierIndices] = pcfitplane(ptCloud, maxDistance);
    
    pause
end
%% Test
ptcfpath = fullfile(selectedFolder,fileList{3});
ptCloud = pcread(ptcfpath);
%roi = [0.45,1.3;3.4,3.9;-1.2,-0.6];
roi = [0.3,1.7;3,4.9;-1.5,0];
roiIndices = findPointsInROI(ptCloud,roi);
ptCloudROI = select(ptCloud,roiIndices);

% Plotting 
f = figure('Name',"Original Point Cloud + ROI");
hold on 
ptCloud.Color = [1,0,0];
ptCloudROI.Color = [0,1,0];
pcshow(ptCloud);
pcshow(ptCloudROI);

% Finding Plane
maxDistance = 0.01;
refNormal = [0,0,1];
maxAngularDistance = 3;
[model, inlierIndices, outlierIndices] = pcfitplane(ptCloudROI, maxDistance,refNormal,maxAngularDistance);
% Getting ptclouds selected and not selected
ptcPlaneIn = select(ptCloudROI,inlierIndices);
ptcPlaneOut = select(ptCloudROI,outlierIndices);

% If using bigger ROI, the ptcPlaneIn is the floor, need to run pcfitplane
% one more time to get the calboard
maxDistance = 0.02;
[model2, inlierIndices2, outlierIndices2] = pcfitplane(ptcPlaneOut, maxDistance);
ptcPlaneIn2 = select(ptcPlaneOut,inlierIndices2);
ptcPlaneOut2 = select(ptcPlaneOut,outlierIndices2);
ptcPlaneIn2.Color = [0,1,0];
ptcPlaneOut2.Color = [1,0,1];

% Coloring
%ptcPlaneOut.Color = [0,1,0];

ptcPlaneIn.Color = [0,0,1];
% Plotting
f = figure("Name","Full + Selected Plane");
hold on 
pcshow(ptCloud);
pcshow(ptcPlaneIn);
pcshow(ptcPlaneIn2);
hold off
f = figure("Name","Selected Plane");
hold on
pcshow(ptcPlaneIn2);
pcshow(ptcPlaneOut2);
hold off


% ptCloud = pcread('data/lidar_1.pcd');
% pcshow(ptCloud);
% boardDim = [456.86, 608.5072];
% detections = detectRectangularPlanePoints(ptCloud,boardDim,"DimensionTolerance",1);

% Try and use the detectRegualarPlanePoints function that takes a list of
% pointcloud file locations and give it the MATLAB equivalent of
% files = ["lidar_{}.pcd".format(i) for i in range(start,stop+1)].  This
% should check if any of the detections worked.

% Also, need to try to restrict the size of the data to get only the board
% to see if that helps