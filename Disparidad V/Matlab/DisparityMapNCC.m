function [ disparityMap, img ] = DisparityMapNCC( left, right, ch, cw )
%DISPARITYMAP Summary of this function goes here
%   Detailed explanation goes here
% left --> imagen izquierda
% right --> imagen derecha
% ch --> tamaño de la ventana de correlacción (height)
% cw --> tamaño de la ventana de correlacción (width)


dimension = size (left);
disparityMap = zeros(dimension(1), dimension(2));

%kernel = fspecial ('sobel');
%kernel = kernel';           % Para gradiente horizontal
%leftEdges = imfilter(left, kernel);
%leftEdges = leftEdges > 8 / 256;
%rightEdges = imfilter(right, kernel);
%rightEdges = rightEdges > 8 / 256;

leftEdges = edge(left, 'sobel');
rightEdges = edge(right, 'sobel');

% -- PRUEBAS

%[dummy1, dummy2, leftGV, leftGH] = edge(left, 'sobel');
%[dummy1, dummy2, rightGV, rightGH] = edge(right, 'sobel');

%leftEdges = leftGH > 8/256;
%rightEdges = rightGH > 8/256;

%imshow(leftEdges)
%max(max(leftEdges))
%figure
%imshow(rightEdges)
%pause

% -- PRUEBAS
margen = 3;

for i = ch+1:dimension(1)-ch
    x = find (leftEdges(i, :) == 1);        % Máximos del gradiente
    for j = 1:length(x)       
        if ((j < length(x)) & (x(j) + margen < x(j + 1)))   % Usar un entorno al máximo de gradiente
            last = x(j) + margen;
        else if (j < length(x))
                last = x(j + 1);           
            else
                last = x(j);
            end
        end

        first = x(j);
        
        for k = first:last
            disparityMap(i, k) = (k - ncc(k, left(i-ch: i+ch, k-cw : k+cw), right(i-ch: i+ch, :)));
        end
    end
end

% -- PRUEBAS
%img = hist(disparityMap, max(max(disparityMap))+ abs(min(min(disparityMap))));
% -- PRUEBAS

img = hist(disparityMap, max(max(disparityMap)));