img1 = imread('/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_080.jpg');
img2 = imread('/home/neztol/doctorado/Datos/imagenesSueltas/IMAGE_081.jpg');
% img1 = imread('/home/neztol/doctorado/Datos/imagenesSueltas/dto1.jpg');
% img2 = imread('/home/neztol/doctorado/Datos/imagenesSueltas/dto2.jpg');

img1 = rgb2gray(img1);
img2 = rgb2gray(img2);

x1 = sum(img1,1);

x2 = sum(img2,1);

x1=x1/max(abs(x1));
x2=x2/max(abs(x2));

model = [ x1; 1:length(x1) ];
% deg= 0.25 * 3.14 / 180;
% data = [ cos(deg) sin(deg) ; -sin(deg) cos(deg) ] * model;
data = [ x2; 1:length(x2) ];

% X = [ 1:length(x1); 1:length(x2); ]';
% Y = [ x1; x2]';

X = [ 1:length(x1); x1 ]';
Y = [ 1:length(x2); x2 ]';

%  Nonrigid Example 2. Coherent Point Drift (CPD).
%  Nonrigid registration of 2D fish point sets with noise and outliers.
%  Full set optioins is explicitelly defined. If you omit some options the
%  default values are used, see help cpd_register.

% delete some points and add outliers.
%X=[X(10:end,:); 0.3*randn(10,2)];
%Y=[Y([1:15 30:end],:); 0.3*randn(10,2)];

% Init full set of options %%%%%%%%%%
opt.method='nonrigid'; % use nonrigid registration
opt.beta=2;            % the width of Gaussian kernel (smoothness)
opt.lambda=8;          % regularization weight

opt.viz=1;              % show every iteration
opt.outliers=0.7;       % use 0.7 noise weight
opt.fgt=0;              % do not use FGT (default)
opt.normalize=1;        % normalize to unit variance and zero mean before registering (default)
opt.corresp=1;          % compute correspondence vector at the end of registration (not being estimated by default)

opt.max_it=100;         % max number of iterations
opt.tol=1e-10;          % tolerance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Transform, C]=cpd_register(X,Y, opt);

figure,cpd_plot_iter(X, Y); title('Before');
figure,cpd_plot_iter(X, Transform.Y, C);  title('After registering Y to X. And Correspondences');
