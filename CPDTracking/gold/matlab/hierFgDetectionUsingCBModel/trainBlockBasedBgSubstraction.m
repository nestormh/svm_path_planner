% This code corresponds to the steps detailed in section II.A and II.B in
% the paper:
%
% Jing-Ming Guo; Chih-Sheng Hsu; , "Hierarchical method for foreground detection using codebook model," 
% Image Processing (ICIP), 2010 17th IEEE International Conference on , vol., no., pp.3441-3444, 26-29 Sept. 2010
% doi: 10.1109/ICIP.2010.5653862
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5653862&isnumber=5648792
%
% Implementation by: Néstor Morales Hernández (nestor@isaatc.ull.es)

function C = trainBlockBasedBgSubstraction(frame, C, M, N, T, lambda, alpha, nu)
% frame=currentFrame;
% C = C;
% M = M;
% N = N;
% T = T;
% lambda = lambdaBlock;
% alpha = alpha;
% nu = nu;

[P,Q, ~] = size(frame);

% A 12-D feature vector is obtained for each block. If it is not matched to the 
% previous codewords in the CodeBook, it is added as a new codeword for
% this block
for i=1:(P/M)
    for j=1:(Q/N)        
        
        if ((M==1) && (N==1))            
            v = [ frame(i,j,1), frame(i,j,2), frame(i,j,3) ];
        else
            block = frame((i -1) * M + 1: i * M, (j -1) * N + 1:j * N, :);
            v = getBlockHistogram(block);
        end
        
        % Fist iteration, foreground by default
        if (isempty(C(i,j).CB))            
            C(i,j).CB = [ C(i,j).CB; v ];
            C(i,j).W = [ C(i,j).W; 1.0/T ];
            C(i,j).K = C(i,j).K + 1;
%             if ((i==1) && (j==1) && ~((M==1) && (N==1)))
%                 display('incrementando k');
%             end
        else                  
            hasMatched = false;
            % ith codework is matched, background
            for k=1:size(C(i,j).CB,1)
                if (matchCodeWords(v, C(i,j).CB(k,:), lambda))                    
                    C(i,j).CB(k,:) = (1-alpha)*C(i,j).CB(k,:) + alpha*v;
                    C(i,j).W(k) = C(i,j).W(k) + 1.0/T;                       
                    
                    hasMatched = true;                    
                end
            end
            % No codework is matched, foreground
            if (~hasMatched)                              
                C(i,j).CB = [ C(i,j).CB; v ];
                C(i,j).W = [ C(i,j).W; 1.0/T ];
                C(i,j).K = C(i,j).K + 1;
%                 if ((i==1) && (j==1) && ~((M==1) && (N==1)))
%                     display('incrementando k');
%                 end
            end
        end          
    end        
end

% Codewords are sorted in descending order using the weights
for i=1:(P/M)
    for j=1:(Q/N)        
        [ C(i,j).W(:), idx ] = sort(C(i,j).W(:), 1, 'descend');        
        C(i,j).CB(:,:)=C(i,j).CB(idx,:);        
        
        sum = 0;
        for k=1:length(C(i,j).W)
            sum = sum + C(i,j).W(k);            
            if (sum > nu)
                break;
            end
        end
        
        C(i,j).W = C(i,j).W(1:k);
        C(i,j).CB = C(i,j).CB(1:k,:);       
        C(i,j).L = k;
        C(i,j).time = zeros(k,1);
    end
end
