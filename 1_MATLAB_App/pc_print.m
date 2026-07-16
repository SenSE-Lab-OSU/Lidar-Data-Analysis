% This script should save a list of point clouds as PNG files 


%% Select File Location
% Run to select which data to use
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

%% Read Files
% Run to load in the pcfiles 
pcList = {lidarFiles.name}';
fprintf("Opening: \n")
for i = 1:size(pcList,1)
    pcfpath = fullfile(selectedFolder,pcList{i});
    fprintf("\t%s\t", pcfpath);
    pcList{i,2} = pcread(pcfpath);
    fprintf("Read\n");
end

%% Select View
% Run to move view to save for all 
fig = figure("name","Temp Display");
pcshow(pcList{1,2});
% While paused move figure to orientation and zoom level
fprintf("Move figure to view you want to save then hit enter\n")
pause()
pos = campos;
close(fig);
% *Theoretically* running view(3);campos(pos); should return figure to this view 

%% Select ROIs
% Run to select ROIs for each PC
figure("name","Temp Display")

for i = 1:size(pcList,1)
    pcList{i,8} = zeros(3,2); % Adds an 8th column for ROIs Columns 2-7 reserved for PCs
    pcshow(pcList{i,2}) % display PC
    ax = gca;
    shg;
    fprintf("Using dataTips, select bottom left ROI, when finished press" + ...
        " enter\n");
    pause();
    point = findobj(ax,'Type','datatip');
    pcList{i,8}(1,1) = point.X;
    pcList{i,8}(2,1) = point.Y;
    pcList{i,8}(3,1) = point.Z;
    shg;
    fprintf("Using dataTips, select Top right ROI, when finished press" + ...
        " enter\n");
    pause();
    point = findobj(ax,'Type','datatip');
    pcList{i,8}(1,2) = point.X;
    pcList{i,8}(2,2) = point.Y;
    pcList{i,8}(3,2) = point.Z;
    
    pcList{i,8} = sort(pcList{i,8},2);
    pcList{i,8}(:,1) = pcList{i,8}(:,1) - 0.2;
    pcList{i,8}(:,2) = pcList{i,8}(:,2) + 0.2;

end

%% Find Planes
% Run to select planes from ROIs
maxDistance = 0.01;
%maxAngularDistance = 3;
for i = 1:size(pcList,1)
    roiIndices = findPointsInROI(pcList{i,2},pcList{i,8});
    pcList{i,3} = select(pcList{i,2},roiIndices); % ROI PC

    [pcList{i,9}, inlierIndices, outlierIndices] = pcfitplane(pcList{i,3}, maxDistance);
    pcList{i,4} = select(pcList{i,3},inlierIndices); % Plane PC from ROI PC

    pcList{i,4}.Color = [0,1,0]; 
    figName = sprintf("Displaying %d",i);
    figure("name",figName)
    hold on
    pcshow(pcList{i,2})
    pcshow(pcList{i,4})

    

end

%% Print PCs
% Run to print Point Clouds.  
fprintf("Select which point clouds to print:\n");
fprintf("Valid selections up to: %d\n",size(pcList,2))
fprintf("1 is file names, DON'T USE\n");
fprintf("2 is Full PC\n3 is ROI\n4 is located plane\n5 is PC " + ...
    "    + ROI\n6 is ROI + Located Plane\n7 is PC + ROI + Located Plane\n");
valSelection = false;
while(~valSelection)
    selection = input("Choose: ");
    if(selection > 1 && selection <= size(pcList,2))
        valSelection = true;
    else
        fprintf("Invalid Selection, ")
    end
end
% Setting File Name
switch selection
    case 2
        saveName = "lidar_pc_full_";
    case 3
        saveName = "lidar_pc_roi_";
    case 4
        saveName = "lidar_pc_plane_";
    case 5
        saveName = "lidar_pc_full_roi_";
    case 6
        saveName = "lidar_pc_roi_plane_";
    case 7
        saveName = "lidar_pc_full_roi_plane_";
end

fig = figure('name','Print Fig');
ax = gca;
firstTime = true;

% Printing Files, could be integrated with previous case statement, this
% way makes it a bit easier to modify file names...less scrolling
for i = 1:size(pcList,1)
    % Generate full savename
    number = sscanf(pcList{i,1},"lidar_%d.pcd");
    saveNameFull = fullfile(fpath, dataFolder(folderSelect).name, saveName+string(number)+".png");

    % Creating Figure
    cla(ax); hold(ax,"on");
    switch selection
        case 2 % Full
            % Plot all - Auto Color
            pcshow(pcList{i,2})
        case 3 % ROI
            pcshow(pcList{i,3})
        case 4 % Plane
            pcshow(pcList{i,4})
        case 5 % PC + ROI
            pcshow(pcList{i,2}.Location,[1,1,0]) % Full - Yellow
            pcshow(pcList{i,3}.Location,[1,0,0]) % ROI - Red
        case 6 % ROI + Plane
            pcshow(pcList{i,3}.Location,[1,0,0]) % ROI - Red
            pcshow(pcList{i,4}.Location,[0,1,0]) % Plane - Green
        case 7 % Full + ROI + Plane
            pcshow(pcList{i,2}.Location,[1,1,0]) % Full - Yellow
            pcshow(pcList{i,3}.Location,[1,0,0]) % ROI - Red
            pcshow(pcList{i,4}.Location,[0,1,0]) % Plane - Green
    end
   
    if(firstTime)
        firstTime = false;
        fprintf("Set view for printing PCs, then press enter when done.\n")
        pause();
        pos = campos;
        shg;
    else
        view(ax,3); axis(ax,'off'); campos(pos);
    end
    %shg
    %pause()
    % Printing Figure
    saveas(gcf,saveNameFull)
    fprintf("saved: %s\n",saveNameFull);

end

%% Blacbody Radiation
T = 300; % Kelvin
c = 3e8; % m/s
h = 6.626e-34; % J*s
v = c/905e-9; % 1/s
K = 1.38e-23; % J/K

2*h*v^3/c^2 * 1/(exp(h*v/(K*T))-1)