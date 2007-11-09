function [ result ] = Obstacles( disparityImg, roadLine )
%OBSTACLES Summary of this function goes here
%   Detailed explanation goes here

    hold off

    aux = disparityImg > 10;        % Umbralizar para eliminar puntos
    
%figure; imshow(aux);    
    [row, column] = size(aux);      % Buscar l�neas de tama�o m�nimo lineMin
    lineMin = 3;
    aux = aux .* [ones(lineMin,column); aux(1:end-lineMin, :)];

    % Obtener las coordenadas y de todos los puntos de la l�nea de "tendencia"
    x = 1:column;
    y = round((roadLine.rho-x*cos(roadLine.theta*pi/180))/sin(roadLine.theta*pi/180))
    for i = 1:column
       aux(y(i):row, i) = 0;
    end
    
    aux = sum(aux(:,2:end));        % Obtener un histograma (se elimina el 1 por ser irrelevante)
    % Debido a esto todas las gr�ficas aparecen con un desplazamiento de
    % una unidad en el eje horizontal.
    
    plot(aux, 'g');
    hold on
    
    kf = [2/8, 4/8, 2/8];
    fil = conv(aux, kf);            % Filtrar para eliminar picos
    fil = fil(2:end-1);
    plot(fil);
    
%    kd = [-1, 1];
%    d = conv(fil, kd);              % Primera derivada por diferenciacion
%    d = d(2:end);
    
%    j = 1;
%    for i=1:length(d)-1
%        if (d(i) ~= 0) && (sign(d(i)) ~= sign(d(i+1)))
%            sing(j) = i;
%            j = j + 1;
%        end
%    end
    
   j = 1;                           % Calcular los m�ximos comparando con su entorno
   for i=2:length(fil)-1
        if (fil(i) >= fil(i - 1)) && (fil(i) > fil(i + 1))
            maximos(j) = i;
            line([i, i], [0, fil(i)], 'color', 'r');
            j = j + 1;
        end
   end
   
   greatest = max(fil(maximos));    % Seleccionar s�lo aquellos m�ximos que sean del tama�o de un porcentaje del mayor
   factor = 0.33;
   
   j = 1;
   for i = 1:length(maximos)
        if fil(maximos(i)) > factor * greatest
            result(j) = maximos(i);
            line([maximos(i), maximos(i)], [0, fil(maximos(i))], 'color', 'c');
            j = j + 1;
        end  
   end
   
   result = result + 1;