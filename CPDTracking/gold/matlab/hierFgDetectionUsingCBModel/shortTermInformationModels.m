% This code corresponds to the steps detailed in section IV in
% the paper:
%
% Jing-Ming Guo; Chih-Sheng Hsu; , "Hierarchical method for foreground detection using codebook model," 
% Image Processing (ICIP), 2010 17th IEEE International Conference on , vol., no., pp.3441-3444, 26-29 Sept. 2010
% doi: 10.1109/ICIP.2010.5653862
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5653862&isnumber=5648792
%
% Implementation by: Néstor Morales Hernández (nestor@isaatc.ull.es)

function [ C, Cs ] = shortTermInformationModels(frame, C, Cs, M, N, t, alpha, lambda, Dadd, Ddelete, DSdelete)
% frame = currentFrame;
% C = C;
% Cs = Cs;
% M = M;
% N = N;
% T = T;
% lambdaBlock = lambdaBlock;
% lambdaPixel = lambdaPixel;
% alpha = alpha;
% nu = nu;
% Dupdate = Dupdate;
% DSdelete = DSdelete;
% Ddelete = Ddelete;
% Dadd = Dadd;
% t=frameNumber - T + 1;

% Constants
BACKGROUND = 0;
FOREGROUND = 1;
HIGHLIGHT = 2;
SHADOW = 3;

[P,Q, ~] = size(frame);

% A 12-D feature vector is obtained for each block. If it is not matched to the 
% previous codewords in the CodeBook, it is added as a new codeword for
% this block. Just blocks marked as FG are used as input.
for i=1:(P/M)
    for j=1:(Q/N)        
        if (C(i,j).fg == BACKGROUND)
            continue;
        end
        
        if ((M==1) && (N==1))            
            v = [ frame(i,j,1), frame(i,j,2), frame(i,j,3) ];
        else
            block = frame((i -1) * M + 1: i * M, (j -1) * N + 1:j * N, :);
            v = getBlockHistogram(block);
        end        
        
        % Fist iteration, foreground by default
        if (isempty(Cs(i,j).CB))           
            Cs(i,j).L = 1;
            Cs(i,j).CB = v;
            Cs(i,j).W = 1.0;
            Cs(i,j).time = t;
        else                  
            hasMatched = false;            
            hasDAdd = false;
            % ith codework is matched, background
            CsNew = struct('CB', [], 'W', [], 'K', 0, 'L', 0, 'time', [], 'fg', BACKGROUND);
            for k=1:size(Cs(i,j).CB,1)                
                if (matchCodeWords(v, Cs(i,j).CB(k,:), lambda))
                    if (t - Cs(i,j).time(k) < DSdelete)
                        CsNew.CB = [ CsNew.CB; (1-alpha)*Cs(i,j).CB(k,:) + alpha*v ];
                        CsNew.W = [ CsNew.W; Cs(i,j).W(k) + 1.0 ];
                        CsNew.time = [ CsNew.time; t ];                        
                        if (Cs(i,j).W(k) + 1.0 > Dadd)
                            hasDAdd = true;
                        end
                    end    
                                            
                    hasMatched = true;                                        
                end
            end
            CsNew.L = size(CsNew.CB, 1);
            Cs(i,j) = CsNew;
            % No codework is matched, foreground
            if (~hasMatched)                              
                Cs(i,j).L = Cs(i,j).L + 1;
                Cs(i,j).CB = [ Cs(i,j).CB; v ];
                Cs(i,j).W = [ Cs(i,j).W; 1.0 ];
                Cs(i,j).time = [ Cs(i,j).time; t ];
            elseif (hasDAdd)
                CNew = struct('CB', [], 'W', [], 'K', 0, 'L', 0, 'time', [], 'fg', BACKGROUND);
                CsNew = struct('CB', [], 'W', [], 'K', 0, 'L', 0, 'time', [], 'fg', BACKGROUND);
                for k=1:C(i,j).L
                    if (C(i,j).time(k) - t < Ddelete)
                        CNew.CB = [ CNew.CB; C(i,j).CB(k,:) ];
                        CNew.W = [ CNew.W; C(i,j).W(k) ];
                        CNew.time = [ CNew.time; C(i,j).time(k) ];
                    end
                end
                for k=1:Cs(i,j).L
                    if (Cs(i,j).W >= Dadd)
                        CNew.CB = [ CNew.CB; Cs(i,j).CB(k,:) ];
                        CNew.W = [ CNew.W; Cs(i,j).W(k) ];
                        CNew.time = [ CNew.time; Cs(i,j).time(k) ];
                    else
                        CsNew.CB = [ CsNew.CB; Cs(i,j).CB(k,:) ];
                        CsNew.W = [ CsNew.W; Cs(i,j).W(k) ];
                        CsNew.time = [ CsNew.time; Cs(i,j).time(k) ];
                    end
                end
                CNew.L = size(CNew.CB, 1);
                CsNew.L = size(CsNew.CB, 1);
                C(i,j) = CNew;
                Cs(i,j) = CsNew;                
            end
        end          
    end        
end