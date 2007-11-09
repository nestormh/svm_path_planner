function [ disparityMap, img ] = DisparityMapSAD( left, right, wsize, maxd )
%DISPARITYMAP Summary of this function goes here
%   Detailed explanation goes here
% left --> imagen izquierda
% right --> imagen derecha
% wsize --> tamaño de la ventana
% maxd --> disparidad máxima


dimension = size (left);
difVol = zeros(dimension(1), dimension(2), maxd);

%[bw, th, gv, gh]=edge(left, 'sobel');
%left = left .* uint8(gv > 4/256);
%left = left .* uint8(bw);
%[bw, th, gv, gh]=edge(right, 'sobel');
%right = right .* uint8(gv > 4/256);
%right = right .* uint8(bw);

%imshow(left)
%figure
%imshow(right)
%pause

convKernel = ones(wsize(1), wsize(2));

%for d=-maxd:maxd                                   % Disparidades positivas y negativas
 for d = 0:maxd                                     % Disparidades sólo positivas
%    if d < 0                                        % Desplazar imagen izquierda
%        leftShift = [zeros(dimension(1), abs(d)), left];
%        leftShift = leftShift(1:dimension(1), 1:dimension(2));
%        rightShift = right;                          
%    else                                    % Desplazar imagen derecha       
            leftShift = left;
            rightShift = [zeros(dimension(1), d), right];
            rightShift = rightShift(1:dimension(1), 1:dimension(2));          
%    end

%    figure
%    imshow(leftShift)
%    figure
%    imshow(rightShift)
%   pause
    
    difVol(:,:,d+1) = abs(leftShift - rightShift);                 % Calcular diferencia SAD (disparidad positiva)
 %   difVol(:,:,d + maxd + 1) = abs(leftShift - rightShift);      % Calcular diferencia SAD (disparidad positiva y negativa)
 %   difVol(:,:,d+1) = (leftShift - rightShift).^2;                 % Calcular diferencia SSD (disparidad positiva)
 %   difVol(:,:,d + maxd + 1) = (leftShift - rightShift).^2;      % Calcular diferencia SSD (disparidad positiva y negativa)

%    corrVol(:,:,d + maxd + 1) = conv2 (difVol(:,:,d + maxd + 1), convKernel);   %(disparidad positiva y negativa)
    corrVol(:,:,d+1) = conv2 (difVol(:,:,d+1), convKernel);                        % disparidad positiva
 end

semiWsize = floor(wsize/2);                                        % Dejar sólo la parte de la convolución correspondiente a desplazar la ventana
corrVolSize = size(corrVol);
corrVol = corrVol(1 + semiWsize(1) : corrVolSize(1) - semiWsize(1),1 + semiWsize(2) : corrVolSize(2) - semiWsize(2),:);
[disparities, disparityMap] = min(corrVol, [], 3);
%disparityMap = disparityMap - maxd - 1;                % Desplazar el origen en el caso de disparidades positivas y negativas

%img = hist(disparityMap', maxd);        % Calculo de la imagen de disparidad "automático" --> NO FUNCIONA BIEN (Cuando son imágenes iguales se ve)
%img = img';                             % Orientar correctamente la imagen de disparidad

histo = zeros (dimension(1), maxd + 1);                       % Calculo de la imagen de disparidad "manual"
for i=1:dimension(1)
    for j=1:dimension(2)
        histo(i, disparityMap(i,j)) = histo(i, disparityMap(i,j)) + 1;
    end
end
img = histo;
%mesh(histo(:,2:max(max(disparityMap))-1))
