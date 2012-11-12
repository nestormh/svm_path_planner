% This code corresponds to the steps detailed in section III.A in
% the paper:
%
% Jing-Ming Guo; Chih-Sheng Hsu; , "Hierarchical method for foreground detection using codebook model," 
% Image Processing (ICIP), 2010 17th IEEE International Conference on , vol., no., pp.3441-3444, 26-29 Sept. 2010
% doi: 10.1109/ICIP.2010.5653862
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5653862&isnumber=5648792
%
% Implementation by: Néstor Morales Hernández (nestor@isaatc.ull.es)

function [ C, F ] = blockBasedFgDetection(frame, C, F, M, N, t, lambdaBlock, lambdaPixel, alpha, Dupdate)
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

% A 12-D feature vector is obtained for each block. It is then matched
% with one of the blocks in the codeword. If not matches are found, then
% it is foreground
for i=1:(P/M)
    for j=1:(Q/N)    
%         i = 28;
%         j = 19;
        block = frame((i -1) * M + 1: i * M, (j -1) * N + 1:j * N, :);
        v = getBlockHistogram(block);
        
        C(i,j).fg = FOREGROUND;
        for k=1:C(i,j).L
            if (matchCodeWords(v, C(i,j).CB(k,:), lambdaBlock))
                C(i,j).CB(k,:) = (1-alpha)*C(i,j).CB(k,:) + alpha*v;
                C(i,j).K = C(i,j).K + 1.0;
                C(i,j).time(k) = t;
                C(i,j).fg = BACKGROUND;
                                
                if (C(i,j).K > Dupdate)
                    for bi=1:M
                        for bj=1:N
                            y = (i-1)*M + bi;
                            x = (j-1)*N + bj;
                            
                            
                            for bk=1:F(y,x).L
                                vPixel = [ frame(y,x,1), frame(y,x,2), frame(y,x,3) ];
                                
                                if (matchCodeWords(vPixel, F(y,x).CB(bk,:), lambdaPixel))
                                    F(y,x).CB(bk,:) = (1-alpha)*F(y,x).CB(bk,:) + alpha*vPixel;                                    
                                    F(y,x).time(bk) = t;
                                    C(i,j).K = 0;
                                end
                            end                                                        
                        end
                    end
                end            
            end
        end
    end
end