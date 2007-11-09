function [ imfilt ] = filtMax( img, wsize )
%FILTMAX Summary of this function goes here
%   Detailed explanation goes here
%   Filtrado Máximo/mínimo, se asigna como valor filtrado al máximo
%   encontrado en la ventana.
%
%   img --> Imagen a filtrar
%   wsize --> Tamaño de la ventana del filtro

    imgsize = size(img);
    imfilt = zeros(imgsize);
    semiwsize = round(wsize/2);
    
    for i = semiwsize(1)+1:imgsize(1)-semiwsize(1)
        for j = semiwsize(2)+1:imgsize(2)-semiwsize(2)
            imfilt(i, j) = min (min (img(i-semiwsize(1):i+semiwsize(1), j-semiwsize(2):j+semiwsize(2))));
        end
    end
   
