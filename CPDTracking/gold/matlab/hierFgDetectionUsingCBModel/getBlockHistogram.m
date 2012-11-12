% This function generates the 12-D vector v used in the section II.A and
% II.B of the paper:
%
% Jing-Ming Guo; Chih-Sheng Hsu; , "Hierarchical method for foreground detection using codebook model," 
% Image Processing (ICIP), 2010 17th IEEE International Conference on , vol., no., pp.3441-3444, 26-29 Sept. 2010
% doi: 10.1109/ICIP.2010.5653862
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5653862&isnumber=5648792
%
% Implementation by: Néstor Morales Hernández (nestor@isaatc.ull.es)

function v = getBlockHistogram(block)
    blockR = block(:,:,1);
    blockG = block(:,:,2);
    blockB = block(:,:,3);

    blockR = blockR(:);
    blockG = blockG(:);
    blockB = blockB(:);

    mu_R = mean(blockR);
    mu_G = mean(blockG);
    mu_B = mean(blockB);

    mu_h_R = mean(blockR(blockR >= mu_R));
    mu_h_G = mean(blockG(blockG >= mu_G));
    mu_h_B = mean(blockB(blockB >= mu_B));

    mu_l_R = mean(blockR(blockR < mu_R));
    mu_l_G = mean(blockG(blockG < mu_G));
    mu_l_B = mean(blockB(blockB < mu_B));

    v = zeros(1,12);
    
    % Red case
    if (length(unique(blockR)) == 1)
        v(1:4) = mu_R;
    else
        if (length(unique(blockR >= mu_R)) == 1)
            v(1:2) = mu_h_R;
        else
            % mu_ht
            if (isempty(blockR(blockR >= mu_h_R)))
                v(1) = mu_h_R;
            else
                v(1) = mean(blockR(blockR >= mu_h_R));  
            end
            % mu_hb
            if (isempty(blockR((((blockR >= mu_R) .* (blockR < mu_h_R))==1))))
                v(2) = mu_h_R;
            else
                v(2) = mean(blockR((((blockR >= mu_R) .* (blockR < mu_h_R))==1))); 
            end
        end
        if (length(unique(blockR < mu_R)) == 1)
            v(3:4) = mu_l_R;
        else
            % mu_lt
            if (isempty(blockR((((blockR >= mu_l_R) .* (blockR < mu_R))==1))))
                v(3) = mu_l_R;
            else
                v(3) = mean(blockR((((blockR >= mu_l_R) .* (blockR < mu_R))==1))); 
            end
            % mu_lb
            if (isempty(blockR(blockR < mu_l_R)))
                v(4) = mu_l_R;
            else
                v(4) = mean(blockR(blockR < mu_l_R));  
            end
        end
    end
    
    % Green Case
    if (length(unique(blockG)) == 1)
        v(5:8) = mu_G;
    else
        if (length(unique(blockG >= mu_G)) == 1)
            v(5:6) = mu_h_G;
        else
            % mu_ht
            if (isempty(blockG(blockG >= mu_h_G)))
                v(5) = mu_h_G;
            else
                v(5) = mean(blockG(blockG >= mu_h_G));  
            end
            % mu_hb
            if (isempty(blockG((((blockG >= mu_G) .* (blockG < mu_h_G))==1))))
                v(6) = mu_h_G;
            else
                v(6) = mean(blockG((((blockG >= mu_G) .* (blockG < mu_h_G))==1))); % mu_hb
            end
        end
        if (length(unique(blockG < mu_G)) == 1)
            v(7:8) = mu_l_G;
        else
            % mu_lt
            if (isempty(blockG((((blockG >= mu_l_G) .* (blockG < mu_G))==1))))
                v(7) = mu_l_G;
            else
                v(7) = mean(blockG((((blockG >= mu_l_G) .* (blockG < mu_G))==1)));
            end
            % mu_lb
            if (isempty(blockG(blockG < mu_l_G)))
                v(8) = mu_l_G;
            else
                v(8) = mean(blockG(blockG < mu_l_G));   
            end
        end
    end
    
    % Blue case
    if (length(unique(blockB)) == 1)
        v(9:12) = mu_B;
    else
        if (length(unique(blockB >= mu_B)) == 1)
            v(9:10) = mu_h_B;
        else
            % mu_ht
            if (isempty(blockB(blockB >= mu_h_B)))
                v(9) = mu_h_B;
            else
                v(9) = mean(blockB(blockB >= mu_h_B));
            end
            % mu_hb
            if (isempty(blockB((((blockB >= mu_B) .* (blockB < mu_h_B))==1))))
                v(10) = mu_h_B;
            else
                v(10) = mean(blockB((((blockB >= mu_B) .* (blockB < mu_h_B))==1)));
            end
        end
        if (length(unique(blockB < mu_B)) == 1)
            v(11:12) = mu_l_B;
        else
            % mu_lt
            if (isempty(blockB((((blockB >= mu_l_B) .* (blockB < mu_B))==1))))
                v(11) = mu_l_B;
            else
                v(11) = mean(blockB((((blockB >= mu_l_B) .* (blockB < mu_B))==1)));
            end
            % mu_lb
            if (isempty(blockB(blockB < mu_l_B)))
                v(12) = mu_l_B;
            else
                v(12) = mean(blockB(blockB < mu_l_B));
            end
        end
    end        

%     figure(1);
%     title('Histograma de v_b^t');
%     bar(v);

end