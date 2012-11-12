paths={'/home/neztol/doctorado/Datos/EstadisticasITER/ManzanillaTripodeLowRes/', '/home/nestor/doctorado/Datos/EstadisticasITER/tripode1LowRes/', '/home/nestor/doctorado/Datos/EstadisticasITER/tripode2LowRes/', '/home/nestor/doctorado/Datos/EstadisticasITER/tripode3LowRes/'};
exts={ '.JPG', '.JPG', '.JPG', '.JPG' };
files={ 'datos.txt', 'datos.txt', 'datos.txt', 'datos.txt' };
errPath='/home/nestor/matlab/workspace/cpd/errorsSurf.txt';
statPath='/home/nestor/matlab/workspace/cpd/statSurf.txt';

for p=1:length(paths)    
    try
        testSurfCPD(char(paths(p)), char(files(p)), char(exts(p)), 0, 'Surf', 1, statPath, errPath)
    catch em
        fid=fopen(errPath, 'a');
        fprintf(fid, '%s\t%f\t%f\t%s\t%s\n', char(paths(p)), -1, -1, 'Surf', getReport(em, 'extended'));
        fclose(fid);
    end
end
