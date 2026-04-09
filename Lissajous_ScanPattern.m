clear; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variables that I think are set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
natFreq = 270; % Hz
natT = 1/natFreq; % s, period
% This is the natural frequency or eigenfrequency for both of the mems
% mirrors.  It appears that the actual frequency of the mirrors are an
% integer multiple of the eigenfrequency






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variables I know you can change %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thetaMaxH = 80; % Degrees - Max horizontal angle of the lidar 
thetaMaxV = 30; % Degrees - Max vertival angle of the lidat
% This angle is centered, so left/top and right/bottom extends 1/2 of max

scanLines = 40; % Horizontal scan lines, each horizontal period=> 2 scanlines
numPeriodsLow = scanLines/2;

duration = 2; % Number of natural period per two scanlines, 
% Blickfeld documentation seems to define the 'frame time' as T.  Then it
% declares the duration of each period as x/natFreq, which sounds like
% saying the frame time = x/T, but I think this is just an unrelated T
freq = natFreq/duration;
omega = 2*pi*freq; % rad/s
T = 1/freq; % seconds - Horizontal Period 
frameTime = T * numPeriodsLow; % seconds - Total time to collect frame

resolution = 1300;
t = linspace(0,frameTime,resolution); % Timescale for the plotting

%%%%%%%%%%%%%%%%%
% Ramp Function %
%%%%%%%%%%%%%%%%%
% This function determines the density of the horizontal scanlines over
% time.  According to the documentation it can change.  However, I am 
% uncertain ~how~ to change it.

flipPoint = 3/4; % Set 0-1
index = floor(flipPoint*resolution);
r = ones(1,resolution);
r(1:index) = 1/flipPoint * t(1:index)/frameTime; % 0-1 from 0-3/4 of frame time
r(index+1:end) = (1-1*t(index+1:end)/frameTime)/(1-t(index)/frameTime); % 1-0 from3/4-end of frame time

% r(1:floor(resolution/2)) = linspace(0,0.5,length(1:floor(resolution/2)));
% r(floor(resolution/2)+1:floor(3/4*resolution)) = linspace(0.5,0.6,length(floor(resolution/2)+1:floor(3/4*resolution)));
% r(floor(3/4*resolution):resolution) = linspace(0.6,1,length(floor(3/4*resolution):resolution));


%%%%%%%%%%%%%%%%%%%%%%%
% The angle functions %
%%%%%%%%%%%%%%%%%%%%%%%

% Horizontal angle
thetaH = thetaMaxH/2 * cos(omega*t);
% Vertical angle
% The documentation defines the mirror phase difference as pi/4, then calls
% the two angle functions as sin,cos omega*t....so I think they really just
% meant the phase difference is pi/2!
thetaV = thetaMaxV/2 * sin(omega*t) .* r;

%% UPSHOT
% It seems like the ability to define different ramp functions would give
% us the opportunity to dynamically change the scan patternand thus
% theoretically get far denser scans of certain ROI. Perhapse in the future
% if humans are located (and denser point clouds are helpful) we can focus
% on them!
close all;
figure; hold on;
plot(t,thetaH,"DisplayName","Horizontal Angle","LineWidth",4);
plot(t,thetaV,"DisplayName",'Vertical Angle','LineWidth',4); 
plot(t,thetaMaxV/2*r,"DisplayName","Ramp Function",'LineStyle',':','Color','k')
legend(); title("Mirror angles over Time");xlabel("Time(seconds)");ylabel("Angle(degrees)");

%%
figure;
p=plot(thetaH,thetaV,'k',"Marker","o","MarkerFaceColor","r","MarkerSize",15);

for i = 1:10:length(t)
  p.MarkerIndices = i;  
  %pause(0.05)
  %drawnow;
  exportgraphics(gca,"scanline_example_squished.gif",append=true);
end

%% Downsampling
natFreq = 270; % Hz
natT = 1/natFreq; % s, period
thetaMaxH = 80; % Degrees - Max horizontal angle of the lidar 
thetaMaxV = 30; % Degrees - Max vertival angle of the lidat
% This angle is centered, so left/top and right/bottom extends 1/2 of max

scanLinesHigh = 512; % Horizontal scan lines, each horizontal period=> 2 scanlines
scanLinesLow = 64;

numPeriodsHigh = scanLinesHigh/2;
numPeriodsLow = scanLinesLow/2;

duration = 2;
freq = natFreq/duration;
omega = 2*pi*freq; % rad/s
T = 1/freq; % seconds - Horizontal Period 
frameTimeHigh =T * numPeriodsHigh;
frameTimeLow = T * numPeriodsLow; % seconds - Total time to collect frame

resolution = 40;
timeHigh = linspace(0,frameTimeHigh,resolution*numPeriodsHigh); % Timescale for the plotting
timeLow  = linspace(0,frameTimeLow,resolution*numPeriodsLow);


flipPoint = 3/4; % Set 0-1
indexHigh = floor(flipPoint*resolution*numPeriodsHigh);
indexLow = floor(flipPoint*resolution*numPeriodsLow);
rHigh = ones(1,resolution*numPeriodsHigh);
rHigh(1:indexHigh) = 1/flipPoint * timeHigh(1:indexHigh)/frameTimeHigh; % 0-1 from 0-3/4 of frame time
rHigh(indexHigh+1:end) = (1-1*timeHigh(indexHigh+1:end)/frameTimeHigh)/(1-timeHigh(indexHigh)/frameTimeHigh); % 1-0 from3/4-end of frame time
rLow  = ones(1,resolution*numPeriodsLow);
rLow(1:indexLow) = 1/flipPoint * timeLow(1:indexLow)/frameTimeLow; % 0-1 from 0-3/4 of frame time
rLow(indexLow+1:end) = (1-1*timeLow(indexLow+1:end)/frameTimeLow)/(1-timeLow(indexLow)/frameTimeLow); % 1-0 from3/4-end of frame time

% Index for downsampling
downTime = zeros(length(timeLow),1);
downSampleBy = scanLinesHigh/scanLinesLow;
j = 1;
for i = 0:numPeriodsLow-1
    %fprintf("Input at %d to %d\n\tfrom %d to %d\n",resolution*(j-1)+1,resolution*j,(i-1+downSampleBy)*resolution+1,(i-1+downSampleBy)*resolution+resolution)
    downTime(resolution*(j-1)+1:resolution*j) = (i*downSampleBy)*resolution+1:(i*downSampleBy)*resolution+resolution;
    j=j+1;
end


thetaHHigh = thetaMaxH/2 * cos(omega*timeHigh);
thetaVHigh = thetaMaxV/2 * sin(omega*timeHigh) .* rHigh;
thetaHLow = thetaMaxH/2 * cos(omega*timeLow);
thetaVLow = thetaMaxV/2 * sin(omega*timeLow) .* rLow;


figure; tiledlayout(3,1);
% Plot High
nexttile; hold on;
plot(timeHigh, thetaHHigh, 'DisplayName', 'Horizontal Angle', 'LineWidth', 2);
plot(timeHigh, thetaVHigh, 'DisplayName', 'Vertical Angle', 'LineWidth', 2);
plot(timeHigh, rHigh*thetaMaxV/2, 'DisplayName', 'Ramp Function * thetaMaxV', 'LineWidth', 2);
legend(); title(sprintf("Mirror angles for %d scanlines",scanLinesHigh)); xlabel('Time (seconds)'); ylabel('Angle (degrees)');
% Plow Low
nexttile; hold on;
plot(timeLow, thetaHLow, 'DisplayName', 'Horizontal Angle', 'LineWidth', 2);
plot(timeLow, thetaVLow, 'DisplayName', 'Vertical Angle', 'LineWidth', 2);
plot(timeLow, rLow*thetaMaxV/2, 'DisplayName', 'Ramp Function * thetaMaxV', 'LineWidth', 2);
legend(); title(sprintf("Mirror angles for %d scanlines",scanLinesLow)); xlabel('Time (seconds)'); ylabel('Angle (degrees)');
% Plot downsampled
nexttile; hold on
thetaHDownsample = zeros(size(thetaHHigh));
thetaHDownsample(downTime) = thetaHHigh(downTime);
thetaVDownsample = zeros(size(thetaVHigh));
thetaVDownsample(downTime) = thetaVHigh(downTime);
plot(timeHigh, thetaHDownsample, 'DisplayName', 'Horizontal Angle', 'LineWidth', 2);
plot(timeHigh, thetaVDownsample, 'DisplayName', 'Horizontal Angle', 'LineWidth', 2);
plot(timeHigh, rHigh*thetaMaxV/2, 'DisplayName', 'Ramp Function * thetaMaxV', 'LineWidth', 2);
legend(); title(sprintf("Mirror angles for %d scanlines downsamped to %d",scanLinesHigh,scanLinesLow)); xlabel('Time (seconds)'); ylabel('Angle (degrees)');

figure;
plot(timeHigh, thetaHDownsample, 'DisplayName', 'Horizontal Angle', 'LineWidth', 2);
legend(); title(sprintf("Mirror angles for %d scanlines downsampled to %d\nzoomed into 1 period",scanLinesHigh,scanLinesLow)); xlabel('Time (seconds)'); ylabel('Angle (degrees)');
xlim([timeHigh(downTime(1)),timeHigh(downTime(resolution))]);


% Plotting scan lines
figure; 
plot(thetaHHigh,thetaVHigh,'k'); xlabel("Horizontal Angle"); ylabel("Vertical Angle"); title(sprintf("Mirror angles for %d scanlines",scanLinesHigh));
figure;
plot(thetaHLow,thetaVLow,'k'); xlabel("Horizontal Angle"); ylabel("Vertical Angle"); title(sprintf("Mirror angles for %d scanlines",scanLinesLow));
figure;
plot(thetaHHigh(downTime),thetaVHigh(downTime),'k'); xlabel("Horizontal Angle"); ylabel("Vertical Angle"); title(sprintf("Mirror angles for %d scanlines downsampled to %d scanlines",scanLinesHigh, scanLinesLow));

%% Error 
errorH = thetaHLow - thetaHDownsample(downTime);
errorV = thetaVLow - thetaVDownsample(downTime);
fprintf("Average angle (degrees) difference " + ...
    "between lower scanlines and downsampled scanlines:\nHorizontal=%.4f" + ...
    "\nVertical=%.4f\n\n",mean(abs(errorH)),mean(abs(errorV)));
fprintf("Max angle (degrees) difference " + ...
    "between lower scanlines and downsampled scanlines:\nHorizontal=%.4f" + ...
    "\nVertical=%.4f\n",max(abs(errorH)),max(abs(errorV)));