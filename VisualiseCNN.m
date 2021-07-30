%% 0
clear; close all; clc;


% netPath = Path to .onnx file to visualise
classesPath = 'CASIA-WebFace-Classes.mat';

load(classesPath,'classes');
net = importONNXNetwork(netPath,'OutputLayerType','classification','Classes',classes);
inputSize = net.Layers(1).InputSize;

%% 1
name = "conv1";
channels = 1:36;
I = deepDreamImage(net,name,channels,'PyramidLevels',1);
figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none')

%% 2
name = "conv2";
channels = 1:36;
I = deepDreamImage(net,name,channels, ...
    'Verbose',false, ...
    'PyramidLevels',1);
figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none')

%% 5
name = "conv5";
channels = 1:6;
I = deepDreamImage(net,name,channels, ...
    'Verbose',false, ...
    "NumIterations",20, ...
    'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none')

%% fully connected
name= "new_fc";
% class names: 0000045 0000099 0000100 0000102 0000103 0000105
% channels = [1 2 3 4 5 6];
% class names: 45 4271632 100 168 207 1595801
channels = [1 10169 3 24 40 10571];
cat = net.Layers(end).Classes(channels);
I = deepDreamImage(net,name,channels, ...
    'Verbose',false, ...
    'NumIterations',100, ...
    'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
title(['Layer ',name,' Features'])