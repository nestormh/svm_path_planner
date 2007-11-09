function [ disparityMap, img ] = DisparityMapSAD( left, right, wsize, maxd )
%DISPARITYMAP Summary of this function goes here
%   Detailed explanation goes here
% left --> imagen izquierda
% right --> imagen derecha
% wsize --> tamaño de la ventana
% maxd --> disparidad máxima



dimension = size (left);

N = 1024;
f = (0 : N - 1) / N;

leftFFT = fft(single(left), N, 2);
rightFFT = fft(single(right), N, 2);

disparityMap = leftFFT;
img = rightFFT;

line = 10;
plot (f, leftFFT(line, :));
hold on
plot (f, rightFFT(line, :), 'r');
hold off
axis ([0, 0.5, -5, 1000])
plot (f, leftFFT(line, :) - rightFFT(line, :));
