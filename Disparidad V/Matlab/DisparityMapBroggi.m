function [ disparityMap, img ] = DisparityMapBroggi( left, right, fsize, ftype, wsize, maxd )
%DISPARITYMAP Summary of this function goes here
%   Detailed explanation goes here
% left --> imagen izquierda
% right --> imagen derecha
% fsize --> tamifa�o del filtro
% ftype --> Tipo de filtro a utilizar
%               0 --> media
%               1 --> gaussiano
%               2 --> mediana
%               3 --> Maximo/minimo
%               4 --> media selectivo
% wsize --> tama�o de la ventana
% maxd --> disparidad m�xima

switch ftype
    case 0
        kfilter = fspecial ('average', fsize);          % Filtro de la media
        left = imfilter(left, kfilter, 'conv');
        right = imfilter(right, kfilter, 'conv');
    case 1    
         kfilter = fspecial ('gaussian', fsize);         % Filtro Gaussiano
        left = imfilter(left, kfilter, 'conv');
        right = imfilter(right, kfilter, 'conv');
    case 2                                                  % Filtro de la mediana
        left = medfilt2(left, fsize);
        right = medfilt2(right, fsize);
    case 3                                                  % Filtro del Max/min
        left = filtMax(left, fsize);
        right = filtMax(right, fsize);
    case 4                                                  % Filtro de la media selectivo
        left = filtSelective(left, fsize);
        right = filtSelective(right, fsize);   
end

%figure; imshow(uint8(left));

dimension = size (left);
difVol = zeros(dimension(1), dimension(2), maxd);

[bw, th, gv, gh] = edge(left, 'sobel');
th
ternarizedLeft = floor(gv) + ceil(gv);
ternarizedLeft = (gv < -th/4) * -1 + (gv > th/4);
[bw, th, gv, gh] = edge(right, 'sobel');
th
ternarizedRight = floor(gv) + ceil(gv);
ternarizedRight = (gv < -th/4) * -1 + (gv > th/4);

left = ternarizedLeft;
right = ternarizedRight;

%convKernel = ones(1, dimension(2));
convKernel = ones(wsize(1), wsize(2));

for d = 0:maxd                                     % Disparidades s�lo positivas
    rightShift = [zeros(dimension(1), d), right];
    rightShift = rightShift(1:dimension(1), 1:dimension(2));          
    difVol(:,:,d+1) = abs(left - rightShift);                      % Calcular diferencia SAD (simple)    
    %    difVol(:,:,d+1) = [zeros(dimension(1),d), abs(left(:,d+1:end) - rightShift(:,d+1:end))];                      % Calcular diferencia SAD (descartando desplazamiento)
 %   difVol(:,:,d+1) = (leftShift - rightShift).^2;                 % Calcular diferencia SSD (disparidad positiva)
    corrVol(:,:,d+1) = conv2 (difVol(:,:,d+1), convKernel);                        % disparidad positiva
end

%wsize = [1, dimension(2)];
semiWsize = floor(wsize/2);                                       % Dejar s�lo la parte de la convoluci�n correspondiente a desplazar la ventana
corrVolSize = size(corrVol);
corrVol = corrVol(1 + semiWsize(1) : corrVolSize(1) - semiWsize(1),semiWsize(2) : corrVolSize(2) - semiWsize(2),:);

%Coger primer m�nimo
[disparities, disparityMap] = min(corrVol, [], 3);


%Coger �ltimo m�nimo

%[disparities, disparityMap] = min(corrVol(:,:,end:-1:1), [], 3);
%disparityMap = maxd - disparityMap + 2;
%disparities = disparities(:,end:-1:1);

histo = zeros (dimension(1), maxd + 1);                       % Calculo de la imagen de disparidad "manual"
for i=1:dimension(1)
    for j=1:dimension(2)
        histo(i, disparityMap(i,j)) = histo(i, disparityMap(i,j)) + 1;
    end
end
img = histo;