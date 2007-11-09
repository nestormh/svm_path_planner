function [ disparityMap, img ] = DisparityMap( left, right, w, cw )
%DISPARITYMAP Summary of this function goes here
%   Detailed explanation goes here
% left --> imagen izquierda
% right --> imagen derecha
% w --> disparidad máxima [-w, w]
% cw --> tamaño de la ventana de correlacción

nDisp = 255;

dimension = size (left);
corrVol = zeros(dimension(1), dimension(2), 2*w+1);

kernel = fspecial ('sobel');
kernel = kernel';           % Para gradiente horizontal
left = imfilter(left, kernel);
right = imfilter(right, kernel)

imshow(left)
figure
imshow(right)
figure

for d=-w:w
    if d < 0                                        % Desplazar imagen derecha
        leftShift = [zeros(dimension(1), abs(d)), left];
            leftShift = leftShift(1:dimension(1), 1:dimension(2));
            rightShift = right;                   
    else if d > 0                                   % Desplazar imagen izquierda
            leftShift = left;
            rightShift = [zeros(dimension(1), abs(d)), right];
            rightShift = rightShift(1:dimension(1), 1:dimension(2));
         else                                        % No desplazar ninguna
            leftShift = left;
            rightShift = right;
         end
    end

    dif = abs(leftShift - rightShift);      % Calcular diferencia

    for i = cw+1:dimension(1)-cw              % Construir el volumen de correlación
        for j = cw+1:dimension(2)-cw
%            indice = [i-cw,i+cw,j-cw,j+cw]
            corrVol (i, j, d + w + 1) = sum(sum(dif(i-cw:i+cw, j-cw:j+cw)));
        end
    end
           
end

for i = 1:dimension(1)          %Determinar el mínimo para cada pixel
    for j = 1:dimension(2)
        [disparityMap(i,j), disparities(i,j)] = min(corrVol(i,j,:));
        disparities(i,j) = disparities(i, j) - w - 1;
    end
end

mesh (disparities)

disparityMap = round(disparityMap / max(max(disparityMap)) * nDisp);

img = zeros (dimension(1), 2*w+1);
%for i = 1:dimension(1)          %Determinar el mínimo para cada pixel
%    for j = 1:dimension(2)
%        img(i,disparities(i,j)) = img(i,map(i,j) + 1) + 1;
%    end
%end

%img = hist(map', 255);

img = hist(disparities', 2*w+1);

%close all
%imshow(left)
%figure
%mesh(map)
%figure
%image(map)

