clc; close all; clear;
cam = load('cameraParams.mat');
cam = cam.cameraParams2.Intrinsics;

% Image
I = imread("C:\Users\rose\OneDrive\Documents\0.Important_Files\2.School\3.Research\LidarDataAnalysis\data\data-02-13-2026-Filtered\img\cam_15.png");
[pts,patternDim] = detectCheckerboardPoints(I);
corners = estimateCheckerboardCorners3d(I,cam,45);
corners = [corners ; corners(1,:)];

figure
cam = plotCamera(Size=0.05,Opacity=0.8);
hold on

%numImages = numel(imageFileNames);
%plotColors = hsv(numImages);
%cornerIdx = [1 2 3 4 1];

plot3(corners(:,1),corners(:,2),corners(:,3),LineWidth=2)
%plot3(cornersCamera(cornerIdx,1,i),cornersCamera(cornerIdx,2,i),cornersCamera(cornerIdx,3,i),Color=plotColors(i,:),LineWidth=2)

set(gca,CameraUpVector=[-1 0 0]);