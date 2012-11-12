clc;
% img1 = imread('/home/nestor/doctorado/Datos/EstadisticasITER/tripode1LowRes/DSC_0555.JPG');
% img2 = imread('/home/nestor/doctorado/Datos/EstadisticasITER/tripode1LowRes/DSC_0561.JPG');
% 
% % figure(1);
% % subplot(1,2,1);
% % imshow(img1);
% % subplot(1,2,2);
% % imshow(img2);
% 
% corrMatch(img1, img2, 40, 3);

paths={'/home/neztol/doctorado/Datos/EstadisticasITER/ManzanillaTripodeLowRes/', '/home/nestor/doctorado/Datos/EstadisticasITER/tripode1LowRes/', '/home/nestor/doctorado/Datos/EstadisticasITER/tripode2LowRes/', '/home/nestor/doctorado/Datos/EstadisticasITER/tripode3LowRes/'};
exts={ '.JPG', '.JPG', '.JPG', '.JPG' };
files={ 'datos.txt', 'datos.txt', 'datos.txt', 'datos.txt' };
errPath='/home/nestor/matlab/workspace/cpd/errorsCorr.txt';
statPath='/home/nestor/matlab/workspace/cpd/statCorr.txt';

for p=1:length(paths)    
    try
        testSurfCPD(char(paths(p)), char(files(p)), char(exts(p)), 1, 'Corr', 1, statPath, errPath)
    catch em
        fid=fopen(errPath, 'a');
        fprintf(fid, '%s\t%f\t%f\t%s\t%s\n', char(paths(p)), -1, -1, 'Corr', getReport(em, 'extended'));
        fclose(fid);
    end
end
