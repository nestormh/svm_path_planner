paths={'/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/'};
exts={ '.jpg' };
files={ 'datos.txt' };
errPath='/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/errorsCorr.txt';
statPath='/home/neztol/doctorado/Datos/Aerop/aerop14EneSinObs/statCorr.txt';

% % Path : path donde están las imágenes de ejemplo
% % path='/home/nestor/doctorado/Datos/EstadisticasITER/tripode2LowRes/';
% currPath = '/home/neztol/doctorado/Datos/EstadisticasITER/ManzanillaTripodeLowRes/';
% % currPath = '/home/neztol/doctorado/Datos/imagenesSueltas/DataSets/';
% % Filepath: fichero con los nombres de las imágenes asociados
% filepath='datos.txt';
% % Ext: Extensión de los ejemplos
% ext='.JPG';
% Parámetros del FAST 
paramFAST=10;
paramCorner=0.2; %0.2;
% Parámetros de distancia entre puntos obtenidos con Canny
paso = 5;
% Show: decide si se muestra al final o no
show = 1;
% Método: 'canny' o 'fast'
method='canny'
% nDim: Número de dimensiones usadas (2 ó 3)
nDim=2; 
% ReadData: Lee en vez de generarlo
readData = 0;
% SaveData: Guarda 
saveData = 0;

% Parámetros del CPD
opt.method='nonrigid_lowrank';
%     opt.lambda = 10;
opt.corresp=1;
%     opt.max_it=300;
%     opt.normalize=0;
%     opt.viz=1;
%     opt.tol=1e-10

%--------------------------------------------------------------------------
% FIN de los parámetros modificables
%--------------------------------------------------------------------------
for p=1:length(paths)    
    try
        testSecuenciaCPD(char(paths(p)), char(files(p)), char(exts(p)), paso, paramCorner, show, method, nDim, readData, saveData, opt, 0, '/tmp/stat.txt', '/tmp/err.txt');
    catch em
        fid=fopen(errPath, 'a');
        fprintf(fid, '%s\t%f\t%f\t%s\t%s\n', char(paths(p)), -1, -1, 'Corr', getReport(em, 'extended'));
        fclose(fid);
    end
end