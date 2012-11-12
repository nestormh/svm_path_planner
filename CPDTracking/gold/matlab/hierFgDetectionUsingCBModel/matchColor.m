% This function checks wether a pixel detected as foreground is a shadow, a
% highlight, or a real foreground, following the function described at
% section III.B of the paper:
%
% Jing-Ming Guo; Chih-Sheng Hsu; , "Hierarchical method for foreground detection using codebook model," 
% Image Processing (ICIP), 2010 17th IEEE International Conference on , vol., no., pp.3441-3444, 26-29 Sept. 2010
% doi: 10.1109/ICIP.2010.5653862
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5653862&isnumber=5648792
%
% Implementation by: Néstor Morales Hernández (nestor@isaatc.ull.es)

function colorModel=matchColor(xp, f, beta, gamma, thetaColor)
% xp = currentFrame(141, 152, :);
% xp = double(xp(:)');
% f = F(141, 152);
% beta = beta;
% gamma = gamma;
% thetaColor = thetaColor

xp = double(xp);

% Constants
BACKGROUND = 0;
FOREGROUND = 1;
HIGHLIGHT = 2;
SHADOW = 3;

colorModel = FOREGROUND;
for i=1:size(f,1)
    fi= double(f.CB(i, :));
    
    mod_fi = sqrt(sum(fi.^2));
    mod_xp = sqrt(sum(xp.^2));
    
    yMax = beta * mod_fi;
    yMin = gamma * mod_fi;
    
%     unit_fi = fi / mod_fi
    
    cross_prod = sum(xp .* fi);
    
%     x_proj = cross_prod * unit_fi / mod_fi
    mod_xp_proj = cross_prod / mod_fi;
    
    dist_xp = sqrt(mod_xp^2- mod_xp_proj^2);
    
    theta_xp = atan(dist_xp / mod_xp_proj);
    
    if ((theta_xp < thetaColor) && (yMin <= mod_xp_proj) && (mod_xp_proj < mod_fi))
        colorModel = SHADOW;
        break;
    elseif ((theta_xp < thetaColor) && (mod_fi <= mod_xp_proj) && (mod_xp_proj < yMax))
        colorModel = HIGHLIGHT;
        break;
    end    
end