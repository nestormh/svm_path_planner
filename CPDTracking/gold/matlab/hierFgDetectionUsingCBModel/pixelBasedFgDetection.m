% This code corresponds to the steps detailed in section III.A in
% the paper:
%
% Jing-Ming Guo; Chih-Sheng Hsu; , "Hierarchical method for foreground detection using codebook model," 
% Image Processing (ICIP), 2010 17th IEEE International Conference on , vol., no., pp.3441-3444, 26-29 Sept. 2010
% doi: 10.1109/ICIP.2010.5653862
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5653862&isnumber=5648792
%
% Implementation by: Néstor Morales Hernández (nestor@isaatc.ull.es)

% function F = pixelBasedFgDetection(frame, C, F, M, N, t, lambdaBlock, lambdaPixel, alpha, beta, gamma, thetaColor, Dupdate)
function F = pixelBasedFgDetection(frame, F, t, lambdaPixel, alpha, beta, gamma, thetaColor)
% frame = currentFrame;
% C = C;
% F=F;
% M = M;
% N = N;
% T = T;
% lambdaBlock = lambdaBlock;
% lambdaPixel = lambdaPixel;
% alpha = alpha;
% nu = nu;
% Dupdate = Dupdate;
% t=frameNumber - T + 1;

% Constants
BACKGROUND = 0;
FOREGROUND = 1;
HIGHLIGHT = 2;
SHADOW = 3;

[P,Q, ~] = size(frame);

% A 3-D feature vector is obtained for each pixel. It is then matched
% with one of the pixels in the codeword. If not matches are found, then
% it is foreground. In this case, it is decided wether it is foreground,
% shadow or highlight
for i=1:P
    for j=1:Q    
        v = [ frame(i,j,1), frame(i,j,2), frame(i,j,3) ];
                
        F(i,j).fg = FOREGROUND;
        for k=1:size(F(i,j),1)            
            if (matchCodeWords(v, F(i,j).CB(k,:), lambdaPixel))
                F(i,j).CB(k,:) = (1-alpha)*F(i,j).CB(k,:) + alpha*v;                
                F(i,j).time(k) = t;
                F(i,j).fg = BACKGROUND;
            end
        end
%         if (C(uint8(i/M + 1), uint8(j/N + 1)).fg == BACKGROUND)
%             F(i,j).fg = BACKGROUND;           
%         end
        if (F(i,j).fg ~= BACKGROUND)
            F(i,j).fg = matchColor(v, F(i,j), beta, gamma, thetaColor);           
        end        
    end    
end