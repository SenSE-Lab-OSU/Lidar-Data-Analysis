% From https://www.mathworks.com/help/lidar/ug/lidar-and-camera-calibration.html

imagePath = fullfile(toolboxdir('lidar'),'lidardata','lcc','HDL64','images');
ptCloudPath = fullfile(toolboxdir('lidar'),'lidardata','lcc','HDL64','pointCloud');
cameraParamsPath = fullfile(imagePath,'calibration.mat');

% Load camera intrinsics.
intrinsic = load(cameraParamsPath);

% Load images using imageDatastore.
imds = imageDatastore(imagePath);
imageFileNames = imds.Files;

% Load point cloud files.
pcds = fileDatastore(ptCloudPath,'ReadFcn',@pcread);
ptCloudFileNames = pcds.Files;

% Square size of the checkerboard.
squareSize = 200;

% Set random seed to generate reproducible results.
rng('default')

[imageCorners3d,checkerboardDimension,dataUsed] = ...
    estimateCheckerboardCorners3d(imageFileNames,intrinsic.cameraParams,squareSize);

% Remove image files that are not used.
imageFileNames = imageFileNames(dataUsed);

% Display checkerboard corners.
%helperShowImageCorners(imageCorners3d,imageFileNames,intrinsic.cameraParams)

% Extract the checkerboard ROI from the detected checkerboard image corners.
roi = helperComputeROI(imageCorners3d,5);

% Filter the point cloud files that are not used for detection.
ptCloudFileNames = ptCloudFileNames(dataUsed);
[lidarCheckerboardPlanes,framesUsed,indices] = ...
    detectRectangularPlanePoints(ptCloudFileNames,checkerboardDimension,ROI=roi);

% Remove ptCloud files that are not used.
ptCloudFileNames = ptCloudFileNames(framesUsed);

% Remove image files.
imageFileNames = imageFileNames(framesUsed);

% Remove 3-D corners from images.
imageCorners3d = imageCorners3d(:,:,framesUsed);

%helperShowCheckerboardPlanes(ptCloudFileNames,indices)

[tform,errors] = estimateLidarCameraTransform(lidarCheckerboardPlanes, ...
    imageCorners3d,intrinsic.cameraParams);