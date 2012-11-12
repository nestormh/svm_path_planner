% This function checks wether two codewords are the same, following the function described
% at the section II.B of the paper:
%
% Jing-Ming Guo; Chih-Sheng Hsu; , "Hierarchical method for foreground detection using codebook model," 
% Image Processing (ICIP), 2010 17th IEEE International Conference on , vol., no., pp.3441-3444, 26-29 Sept. 2010
% doi: 10.1109/ICIP.2010.5653862
% URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5653862&isnumber=5648792
%
% Implementation by: Néstor Morales Hernández (nestor@isaatc.ull.es)

function matches=matchCodeWords(rSource, rCodeword, lambda)
    d = double(rSource) - double(rCodeword);   

    aux=(d*d')/length(d);    

    matches = aux < (lambda * lambda);
end