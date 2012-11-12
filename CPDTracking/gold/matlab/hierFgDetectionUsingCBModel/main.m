% params = struct('size', [280 360], ...
%                     'saveResults', true, ...                    
%                     'show', false, ...  
%                     'M', 20, ... 
%                     'N', 20, ...
%                     'T', 1);
%                 
% % pathBase = '/Datos/CPD/campus1/';
% % minFrame = 1;
% % maxFrame = 2000;
% hierFgDetection('/Datos/CPD/campus1/', 1, 2000, params);  
% 
% return;

% Test for jonayParking1
params = struct('T', 57, 'size', [240 320], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/jonayParking1/', 0, 262, params);

% Test for jonayParking2
params = struct('T', 97, 'size', [240 320], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/jonayParking2/', 0, 767, params);

% Test for jonayParking3
params = struct('T', 51, 'size', [240 320], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/jonayParking3/', 0, 395, params);

% Test for campus1
params = struct('T', 50, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/campus1/', 1, 2000, params);

% Test for campus2
params = struct('T', 50, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/campus2/', 1, 2000, params);

% Test for campus3
params = struct('T', 50, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/campus3/', 1, 2000, params);

% Test for campus4
params = struct('T', 1, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/campus4/', 1, 5884, params);

% Test for campus5
params = struct('T', 50, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/campus5/', 1, 5884, params);

% Test for campus6
params = struct('T', 1, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/campus6/', 1, 5884, params);

% Test for passage1
params = struct('T', 32, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/passage1/', 1, 2500, params);

% Test for passage2
params = struct('T', 32, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/passage2/', 1, 2500, params);

% Test for passage3
params = struct('T', 1, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/passage3/', 1, 2500, params);

% Test for passage4
params = struct('T', 1, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/passage4/', 1, 2500, params);

% Test for terrace1
params = struct('T', 48, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/terrace1/', 1, 5010, params);

% Test for terrace2
params = struct('T', 48, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/terrace2/', 1, 5010, params);

% Test for terrace3
params = struct('T', 48, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/terrace3/', 1, 5010, params);

% Test for terrace4
params = struct('T', 48, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/terrace4/', 1, 5010, params);

% Test for terrace5
params = struct('T', 48, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/terrace5/', 1, 4480, params);

% Test for terrace6
params = struct('T', 48, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/terrace6/', 1, 4480, params);

% Test for terrace7
params = struct('T', 1, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/terrace7/', 1, 4480, params);

% Test for terrace8
params = struct('T', 48, 'size', [280 360], 'saveResults', true, 'show', false, 'M', 20, 'N', 20);
hierFgDetection('/Datos/CPD/terrace8/', 1, 4480, params);
