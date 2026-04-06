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
numPeriods = scanLines/2;

duration = 2; % Number of natural period per two scanlines, 
% Blickfeld documentation seems to define the 'frame time' as T.  Then it
% declares the duration of each period as x/natFreq, which sounds like
% saying the frame time = x/T, but I think this is just an unrelated T
freq = natFreq/duration;
omega = 2*pi*freq; % rad/s
T = 1/freq; % seconds - Horizontal Period 
frameTime = T * numPeriods; % seconds - Total time to collect frame

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

figure;
p=plot(thetaH,thetaV,'k',"Marker","o","MarkerFaceColor","r","MarkerSize",15);
for i = 1:10:length(t)
  p.MarkerIndices = i;  
  %pause(0.05)
  %drawnow;
  exportgraphics(gca,"scanline_example.gif",append=true);
end