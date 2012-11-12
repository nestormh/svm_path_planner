function t = saveImgs(currPath, index)

hfigs = get(0, 'children')                          %Get list of figures

for m = 1:length(hfigs)
    figure(hfigs(m)) 
    filename = strcat(currPath, 'out/', num2str(m), '/Imagen', num2str(index));
    
    saveas(hfigs(m), [filename '.fig']) %Matlab .FIG file
%     saveas(hfigs(m), [filename '.emf']) %Windows Enhanced Meta-File (best for powerpoints)
    saveas(hfigs(m), [filename '.png']) %Standard PNG graphics file (best for web)
%     eval(['print -depsc2 ' filename])   %Enhanced Postscript (Level 2 color) (Best for LaTeX documents)
end

t=1;