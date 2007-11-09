function disparity = ncc( index, template, frame )

templateSize = size(template);
cw = floor(templateSize(2)/2);
N = templateSize(1) * templateSize(2);

frameSize = size(frame);

maxD = floor ((frameSize(2) / 3) / 2);   % Valor m�ximo para la disparidad
%maxD = 500;

mt = mean(mean(template));      % Media del patr�n
vart = sum(sum(template .* template)) / N - (mt * mt);  % Varianza del patr�n

taux = template - mt;

% Recorrido a realizar
if ((index - maxD - cw) > 0)
    first = index - maxD;   
else
    first = 1 + cw;
end

if (index + maxD + cw < frameSize(2))
    last = index + maxD;
else
    last = frameSize(2) - cw;
end

for i = first:last
    window = frame(:,i-cw:i+cw);                            % Subventana a analizar
    mf = mean(mean(window));                                % Media
    varf = sum(sum(window .* window)) / N - (mf * mf);      % Varianza
    correlation (i) = (sum(sum(template .* window)) / N - (mf * mt)) / (varf * vart);
end

%    [size(correlation), first, last]
[dummy, disparity] = max(correlation(first:last));      % Disparidad = Posici�n del m�ximo valor de correlaci�n
disparity = disparity + first;
