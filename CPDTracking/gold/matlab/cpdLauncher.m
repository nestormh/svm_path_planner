matlabpool;

opt = struct( ...
    'method', 'nonrigid_lowrank', ...
    'corresp', 1, ...
    'normalize', 1, ...
    'max_it', 150, ...
    'tol', 1e-05, ...
    'viz', 1, ...
    'outliers', 0.1, ...
    'fgt', 2, ...
    'rot', 0, ...
    'scale', 0, ...
    'beta', 2, ...
    'lambda', 3, ...
    'nDims', 3, ...
    'distThresh', 0.01, ...    
    'saveOutput', 0, ...
    'basePath', '', ...
    'sequenceName', '', ...
    'iteration', 0 ...
    );

% Reading params
fid = fopen('/tmp/CPD/paramsCPD.txt');
fscanf(fid, '%s = ', 1); 
opt.method = fscanf(fid, '%s', 1);
fscanf(fid, '%s = ', 1);
opt.corresp = fscanf(fid, '%d', 1); 
fscanf(fid, '%s = ', 1);
opt.normalize = fscanf(fid, '%d', 1); 
fscanf(fid, '%s = ', 1);
opt.max_it = fscanf(fid, '%d', 1); 
fscanf(fid, '%s = ', 1);
opt.tol = fscanf(fid, '%lf', 1); 
fscanf(fid, '%s = ', 1);
opt.viz = fscanf(fid, '%d', 1); 
fscanf(fid, '%s = ', 1);
opt.outliers = fscanf(fid, '%lf', 1); 
fscanf(fid, '%s = ', 1);
opt.fgt = fscanf(fid, '%d', 1); 
fscanf(fid, '%s = ', 1);
opt.rot = fscanf(fid, '%d', 1); 
fscanf(fid, '%s = ', 1);
opt.scale = fscanf(fid, '%d', 1); 
fscanf(fid, '%s = ', 1);
opt.beta = fscanf(fid, '%lf', 1); 
fscanf(fid, '%s = ', 1);
opt.lambda = fscanf(fid, '%lf', 1); 
fscanf(fid, '%s = ', 1);
opt.nDims = fscanf(fid, '%d', 1); 
fscanf(fid, '%s = ', 1);
opt.distThresh = fscanf(fid, '%lf', 1);
fscanf(fid, '%s = ', 1);
opt.saveOutput = fscanf(fid, '%d', 1); 
fscanf(fid, '%s = ', 1); 
opt.basePath = fscanf(fid, '%s', 1);
fscanf(fid, '%s = ', 1); 
opt.sequenceName = fscanf(fid, '%s', 1);
fscanf(fid, '%s = ', 1);
opt.iteration = fscanf(fid, '%d', 1);

fscanf(fid, '%s = ', 1);
nOldPoints = fscanf(fid, '%d', 1);
fscanf(fid, '%s = ', 1);
nNewPoints = fscanf(fid, '%d', 1); 
fclose(fid);

% Read data
fid = fopen('/tmp/CPD/dataCPD.txt');
oldPoints = zeros(nOldPoints, 6);
newPoints = zeros(nNewPoints, 6);
for i=1:nOldPoints
    oldPoints(i,:) = fscanf(fid, '%f ', [ 1 6 ]);
end
for i=1:nNewPoints
    newPoints(i,:) = fscanf(fid, '%f ', [ 1 6 ]);
end
fclose(fid);

[Transform, C]=cpd_register(oldPoints(:, 1:opt.nDims), newPoints(:, 1:opt.nDims), opt);

output = sprintf('iterations = %d\n', Transform.iter);
output = [ output, sprintf('method = %s\n', opt.method) ];
output = [ output, sprintf('corresp = %d\n', opt.corresp) ];
output = [ output, sprintf('normalize = %d\n', opt.normalize) ];
output = [ output, sprintf('max_it = %d\n', opt.max_it) ];
output = [ output, sprintf('tol = %lf\n', opt.tol) ];
output = [ output, sprintf('viz = %d\n', opt.viz) ];
output = [ output, sprintf('outliers = %lf \n', opt.outliers) ];
output = [ output, sprintf('fgt = %d\n', opt.fgt) ];
output = [ output, sprintf('rot = %d\n', opt.rot) ];
output = [ output, sprintf('scale = %d\n', opt.scale) ];
output = [ output, sprintf('beta = %d\n', opt.beta) ];
output = [ output, sprintf('lambda = %d\n', opt.lambda) ];
output = [ output, sprintf('nDims = %d\n', opt.nDims) ];
output = [ output, sprintf('distThresh = %s\n', opt.distThresh) ];
output = [ output, sprintf('saveOutput = %d\n', opt.saveOutput) ];
output = [ output, sprintf('basePath = %s\n', opt.basePath) ];
output = [ output, sprintf('sequenceName = %s\n', opt.sequenceName) ];
output = [ output, sprintf('iteration = %d\n', opt.iteration) ];

output = [ output, sprintf('%d\n', length(oldPoints)) ];
for i=1:length(oldPoints)
    point=oldPoints(i, :);
    output = [ output, sprintf('%d\t%d\t%d\t%d\t%d\t%d\n', point(1), point(2), point(3), point(4), point(5), point(6)) ];
end

output = [ output, sprintf('%d\n', length(newPoints)) ];
for i=1:length(newPoints)
    point=newPoints(i, :);
    output = [ output, sprintf('%d\t%d\t%d\t%d\t%d\t%d\n', point(1), point(2), point(3), point(4), point(5), point(6)) ];
end

output = [ output, sprintf('%d\n', length(Transform.Y)) ];
for i=1:length(Transform.Y)
    point=newPoints(i, :);
    pointT=Transform.Y(i, :);
    output = [ output, sprintf('%d\t%d\t%d\t%d\t%d\t%d\n', pointT(1), pointT(2), pointT(3), point(4), point(5), point(6)) ];
end

output = [ output, sprintf('%d\n', length(C)) ];
for i=1:length(C)    
    output = [ output, sprintf('%d\n', C(i)) ];
end

%     point2T=oldPoints(C(i), :);
% for i=1:length(C)
%     point1=newPoints(i, :);
%     point2=oldPoints(i, :);
%     point2T=oldPoints(C(i), :);
%     
% %     dist = norm(point1 - point2);    
% %     
% %     if (dist < opt.distThresh)
% %         output = [ output, sprintf('%d\n', C(i)) ];
% %         output = [ output, sprintf('%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n', point1(1), point1(2), point1(3), point1(4), point1(5), point1(6), point2(1), point2(2), point2(3), point2(4), point2(5), point2(6)) ];
%         output = [ output, sprintf('%d\t%d\t%d\t%d\t%d\t%d\t', point1(1), point1(2), point1(3), point1(4), point1(5), point1(6)) ];
%         output = [ output, sprintf('%d\t%d\t%d\t%d\t%d\t%d\t', point2(1), point2(2), point2(3), point2(4), point2(5), point2(6)) ];
%         output = [ output, sprintf('%d\t%d\t%d\t%d\t%d\t%d\n', point2T(1), point2T(2), point2T(3), point2T(4), point2T(5), point2T(6)) ];
% %     else        
% %         output = [ output, sprintf('-1\t-1\t-1\t-1\t-1\t-1\n') ];
% % %         output = [ output, sprintf('-1\n') ];
% %     end
% end

output=output(1:end-1);
fid = fopen('/tmp/CPD/outputCPD.txt', 'w');
fprintf(fid, output);
fclose(fid);

if (opt.saveOutput)
    if (~exist(sprintf('%s/%s/corresp', opt.basePath, opt.sequenceName), 'dir'))
        mkdir(sprintf('%s/%s', opt.basePath, opt.sequenceName),'corresp')
    end

    fid = fopen(sprintf('%s/%s/corresp/corresp%d.txt', opt.basePath, opt.sequenceName, opt.iteration), 'w');
    fprintf(fid, output);
    fclose(fid);
end

matlabpool close;