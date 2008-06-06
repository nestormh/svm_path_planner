function [ disparityMap, img ] = Captura( cam_izq, cam_der, file_number )
%CAPTURA Summary of this function goes here
%   Detailed explanation goes here

    izq = getsnapshot(cam_izq);
    imwrite(izq, strcat('izquierda',int2str(file_number),'.jpg'), 'jpg');
    izq = rgb2gray(izq);
    figure; 
    imshow(izq);
    der = getsnapshot(cam_der);
    imwrite(der, strcat('derecha',int2str(file_number),'.jpg'), 'jpg');
    der = rgb2gray(der);
    figure; 
    imshow(der);
    
%    [disparityMap, img] = DisparityMapBroggi (izq, der, [1, 9], 50);
%    [disparityMap, img] = DisparityMapBroggi(izq, der, [3, 3],0,[1,9],70);
%    figure;
%    imshow(img);