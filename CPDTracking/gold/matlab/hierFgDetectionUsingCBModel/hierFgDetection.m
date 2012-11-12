% This code corresponds to the method described in the paper:
%
% Jing-Ming Guo; Chih-Sheng Hsu; , "Hierarchical method for foreground detection using codebook model," 
% Image Processing (ICIP), 2010 17th IEEE International Conference on , vol., no., pp.3441-3444, 26-29 Sept. 2010
% doi: 10.1109/ICIP.2010.5653862
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5653862&isnumber=5648792
%
% Implementation by: Néstor Morales Hernández (nestor@isaatc.ull.es)

function hierFgDetection(pathBase, minFrame, maxFrame, params)

% Constants
BACKGROUND = 0;
FOREGROUND = 1;
HIGHLIGHT = 2;
SHADOW = 3;

% Params are assigned
defparams = struct('size', [280 360], ...
                    'saveResults', false, ...                    
                    'show', false, ...  
                    'lambdaBlock', 5, ... % Performance-guided parameters
                    'lambdaPixel', 6, ...
                    'nu', 0.7, ...
                    'thetaColor', 3.0, ...
                    'beta', 1.15, ...
                    'gamma', 0.72, ...
                    'Dupdate', 3, ...                    
                    'alpha', 0.05, ... % Application-guided parameters
                    'Dadd', 100, ...
                    'DSdelete', 200, ...
                    'Ddelete', 200, ...                    
                    'M', 20, ... % Other parameters
                    'N', 20, ...
                    'T', 50, ...
                    'CameraStr', 'LS');

names = fieldnames(params);
finalParams = defparams;
for i=1:length(names)       
    finalParams = setfield(finalParams, char(names(i)), getfield(params, char(names(i))));
end

mySize = finalParams.size;
saveResults = finalParams.saveResults;
show = finalParams.show;
lambdaBlock = finalParams.lambdaBlock;
lambdaPixel = finalParams.lambdaPixel;
nu = finalParams.nu;
thetaColor = finalParams.thetaColor;
beta = finalParams.beta;
gamma = finalParams.gamma;
Dupdate = finalParams.Dupdate;
alpha = finalParams.alpha;
Dadd = finalParams.Dadd;
DSdelete = finalParams.DSdelete;
Ddelete = finalParams.Ddelete;
M = finalParams.M;
N = finalParams.N;
T = finalParams.T;
                
sequence = fullfile(pathBase,'frames');
if (saveResults)
    resultsPath = fullfile(pathBase,'resultsHierachical');
    blockBasePath=fullfile(resultsPath,'block');
    pixelBasePath=fullfile(resultsPath,'pixel');
    bgBasePath=fullfile(resultsPath,'background');

    if (~exist(resultsPath, 'dir'))
        mkdir(resultsPath);
    end
    if (~exist(blockBasePath, 'dir'))
        mkdir(blockBasePath);
    end
    if (~exist(pixelBasePath, 'dir'))
        mkdir(pixelBasePath);
    end
    if (~exist(bgBasePath, 'dir'))
        mkdir(bgBasePath);
    end
end

% Start of the main program
currentFrame=imresize(imread(fullfile(sequence,sprintf('image%07d%s.bmp', minFrame, params.CameraStr))), mySize);
P = size(currentFrame, 1);
Q = size(currentFrame, 2);

if ((mod(P,M) ~= 0) || (mod(Q,N) ~= 0))
    display(sprintf('Image size (%d,%d) is not valid', P, Q));
    return;
end

C(1:P/M, 1:Q/N) = struct('CB', [], 'W', [], 'K', 0, 'L', 0, 'time', [], 'fg', BACKGROUND);
F(1:P, 1:Q) = struct('CB', [], 'W', [], 'K', 0, 'L', 0, 'time', [], 'fg', BACKGROUND);
Cs(1:P/M, 1:Q/N) = struct('CB', [], 'W', [], 'K', 0, 'L', 0, 'time', [], 'fg', BACKGROUND);
Fs(1:P, 1:Q) = struct('CB', [], 'W', [], 'K', 0, 'L', 0, 'time', [], 'fg', BACKGROUND);

display('****************************************');
display('Starting training phase...');
for frameNumber=minFrame:T        
    display(sprintf('Training frame %d of %d ...', frameNumber, T));
    currentFrame=imresize(imread(fullfile(sequence,sprintf('image%07d%s.bmp', frameNumber, params.CameraStr))), mySize);
    
    C = trainBlockBasedBgSubstraction(currentFrame, C, M, N, T, lambdaBlock, alpha, nu);
    F = trainBlockBasedBgSubstraction(currentFrame, F, 1, 1, T, lambdaPixel, alpha, nu);                

end
if (saveResults)    
    save fullfile(resultsPath,'training.mat');
end
display('****************************************');

% load 'afterTraining.mat';

display('****************************************');
display('Starting testing phase...');
for frameNumber=(T+1):maxFrame
    display(sprintf('Testing frame %d of %d ...', frameNumber, maxFrame));
    
    currentFrame=imresize(imread(fullfile(sequence,sprintf('image%07d%s.bmp', frameNumber, params.CameraStr))), mySize);
    t = frameNumber - T;
    
    [ C, F ] = blockBasedFgDetection(currentFrame, C, F, M, N, frameNumber - T, lambdaBlock, lambdaPixel, alpha, Dupdate);    
    F = pixelBasedFgDetection(currentFrame, F, frameNumber - T, lambdaPixel, alpha, beta, gamma, thetaColor);    
    [ C, Cs ] = shortTermInformationModels(currentFrame, C, Cs, M, N, t, alpha, lambdaBlock, Dadd, Ddelete, DSdelete);
    [ F, Fs ] = shortTermInformationModels(currentFrame, F, Fs, 1, 1, t, alpha, lambdaPixel,  Dadd, Ddelete, DSdelete);
            
    blockFg = zeros(P,Q);
    for i=1:(P/M)
        for j=1:(Q/N)
            if (C(i,j).fg == FOREGROUND)
                blockFg((i-1)*M+1:i*M+1,(j-1)*N+1:j*N+1) = 255;
            end
        end
    end
    
    pixelBg = uint8(zeros(P,Q,3));
    for i=1:P
        for j=1:Q
            pixelBg(i,j,1) = F(i,j).CB(1,1);
            pixelBg(i,j,2) = F(i,j).CB(1,2);            
            pixelBg(i,j,3) = F(i,j).CB(1,3);
        end
    end
    
    pixelFg = zeros(P,Q);
    for i=1:P
        for j=1:Q
            if (F(i,j).fg ~= BACKGROUND)
                pixelFg(i,j) = F(i,j).fg;
            end
        end
    end
    pixelFg = label2rgb(pixelFg);
    
    if (saveResults)
        blockPath=fullfile(blockBasePath,sprintf('Image%d.png', t));
        pixelPath=fullfile(pixelBasePath,sprintf('Image%d.png', t));
        bgPath=fullfile(bgBasePath,sprintf('Image%d.png', t));
        
        imwrite(blockFg, blockPath);
        imwrite(pixelFg, pixelPath);
%         imwrite(pixelBg, bgPath);                
    end
    
    
    if (show)
        figure(1);
        subplot(2,2,1);
        imshow(currentFrame);
        subplot(2,2,2);
        imshow(pixelBg);
        subplot(2,2,3);
        imshow(blockFg);
        subplot(2,2,4);
        imshow(pixelFg);
        
        waitforbuttonpress;
    end

end
if (saveResults)
    save fullfile(resultsPath,'testing.mat');
end
display('****************************************');


