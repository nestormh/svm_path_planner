function t = testSecuenciaCPD(currPath, filepath, ext, param, paramCorner, show, method, nDim, readData, saveData, opt, stat, statPath, errPath)
%testCPDComentado Empareja imágenes usando Coherent Point Drift
%   t = testCPDComentado(currPath, filepath, ext, param, paramCorner, param,
%   show, method, nDim, readData, saveData, opt)
%   
% currPath : path donde están las imágenes de ejemplo
% filepath: fichero con los nombres de las imágenes asociados
% ext: Extensión de los ejemplos (por ejemplo: '.JPG'). Cuidado con las
% mayúsculas y las minúsculas, y el punto inicial
% param: Parámetro del método FAST/Parámetro de distancia entre puntos
% obtenidos con Canny (depende del método)
% paramCorner: Parámetro del método de eliminación de la máxima supresión
% (relacionado con FAST)
% show: decide si se muestra al final o no
% method: 'canny' o 'fast'
% nDim: Número de dimensiones usadas (2 ó 3)
% readData: Lee en vez de generar los emparejamientos con CPD (debe haber
% sido aplicado el CPD anteriormente)
% saveData: Guarda los resultados
% opt: Parámetros del CPD

t=0;
opt.viz=show;

% Abrimos la imagen "central" y la ponemos en escala de grises
% dist: distancia a la imagen central
filepath=[currPath,filepath]

[imgName] = textread(filepath, '%s');
    
imgName(:)=strcat(currPath,imgName(:),ext);

% Recorremos todas las imágenes del fichero
for i=2:length(imgName)        
    try
        imName = char(imgName(i - 1))
        img1=imread(imName);
        img1=double(rgb2gray(img1));
        
        imName = char(imgName(i))
        img2=imread(imName);
        img2=double(rgb2gray(img2));
              
        % Si estamos usando canny, obtenemos las esquinas iniciales
        if (strmatch('canny', method))
            borders1=edge(img1, 'canny');

            corners1 = ones(1,2);
            index=1;
            pos = 0;
            for a=1:size(borders1,1)
                for b=1:size(borders1,2)
                    if (borders1(a,b)==1)
                        pos = pos + 1;
                        if (mod(pos, param) ~= 0) 
                            continue;            
                        end
                        corners1(index,1) = b;
                        corners1(index,2) = a;
                        index = index + 1;
                    end
                end
            end
        end

        % Si estamos usando FAST, obtenemos las esquinas iniciales
        if (strmatch('fast', method))
            corners1 = fast_corner_detect_9(img1, param);
            corners1 = fast_nonmax(img1, param, corners1);
            cout=corner(img1, [], [], [], paramCorner);
            corners1=[cout(:,2), cout(:,1)];
        end

        % Creamos la primera ventana
        if (show)
            figure(4);
            subplot(1,2,1);
            image(img1/4);
            axis image;
            colormap(gray);
            hold on;
            plot(corners1(:,1), corners1(:,2), 'r.');
            hold off;
        end
        
        % Si estamos usando canny, obtenemos las esquinas iniciales
        if (strmatch('canny', method))
            
            borders2=edge(img2, 'canny');
            
            corners2 = ones(1,2);
            index=1;
            pos = 0;
            for a=1:size(borders2,1)
                for b=1:size(borders2,2)
                    if (borders2(a,b)==1)
                        pos = pos + 1;
                        if (mod(pos, param) ~= 0)
                            continue;
                        end
                        corners2(index,1) = b;
                        corners2(index,2) = a;
                        index = index + 1;
                    end
                end
            end
        end
        
        % Si estamos usando FAST, obtenemos las esquinas iniciales
        if (strmatch('fast', method))
            corners2 = fast_corner_detect_9(img2, param);
            corners2 = fast_nonmax(img2, param, corners2);
            cout=corner(img2, [], [], [], paramCorner);
            corners2=[cout(:,2), cout(:,1)];
        end
        
        if (show)
            figure(4);
            subplot(1,2,2);
            image(img2/4);
            axis image;
            colormap(gray);
            hold on;
            plot(corners2(:,1), corners2(:,2), 'r.');
            hold off;
        end
        
        % En caso de estar usando las 3 dimensiones, añadimos la tercera
        % dimensión
        if (nDim == 3)
            corners1 = [corners1(:,1), corners1(:, 2), corners1(:,2)];
            corners2 = [corners2(:,1), corners2(:, 2), corners2(:,2)];
            for j=1:length(corners1)
                %             corners1(j,:)=[corners1(j,1), corners1(j,2), img1(corners1(j,2), corners1(j,1))];
                corners1(j,3)=img1(corners1(j,2), corners1(j,1));
            end
            for j=1:length(corners2)
                corners2(j,3)=img2(corners2(j,2), corners2(j,1));
                %             corners2(j,:)=[corners2(j,1), corners2(j,2), img2(corners2(j,2), corners2(j,1))];
            end
        end
        
        % Mostramos la evolución del CPD
        corners1 = corners1(:,1:2);
        corners2 = corners2(:,1:2);
        
        if (show)
            figure(1);            
        end

        if (readData == 0)
            try
                [Transform, C]=cpd_register(corners1,corners2, opt);
            catch me
                me.identifier
                if (stat)
                    fid=fopen(statPath, 'a');
                    fprintf(fid, '%s\t%f\t%f\t%f\t%s\t%d\t%s\t%d\t%d\t%d\n', char(currPath), dist(i), ang(i), param, method, nDim, opt.method, 0, 0, 0);
                    fclose(fid);
                end
            end
        end
        
        if (show)
            figure(5);
            subplot(1,2,1);
            image(img1/4);
            axis image;
            colormap(gray);

            subplot(1,2,2);
            image(img2/4);
            axis image;
            colormap(gray);
        end
        
        % Si el punto resultante calculado está distante del inicial, se
        % descarta (ver la parte de pintado)
        if (readData == 0)
            distVect=sqrt((corners1(C(:), 1) - Transform.Y(:, 1)).^2 + (corners1(C(:), 2) - Transform.Y(:, 2)).^2);
        end
        
        % Obtiene el nombre del fichero con los datos
        matName=strcat(imgName(i - 1), '.', imgName(i), '.', method, num2str(nDim), 'D', 'param', num2str(param), '.mat');
        
        % Lee los resultados si no desean ser generados
        if (readData)
            clear corners1 corners2 C distVect Transform;
            load(char(matName));
        end
        
        % Guarda los resultados para poderlos visualizar más adelante
        if (saveData)
            save(char(matName),'corners1','corners2','distVect','C','Transform');
        end
        
        % Con un RANSAC terminamos de eliminar outliers
        x1 = [corners1(C(:),1), corners1(C(:),2), ones(length(C),1)]';
        x2 = [corners2(:,1), corners2(:,2), ones(length(C),1)]';
        [F, inliers] = ransacfitfundmatrix(x1, x2, 0.001);
        
        % Obtenemos la diferencia de imágenes, usando los puntos seleccionados
        tmp1=ones(2,2);
        tmp2=ones(2,2);
        for j=1:length(C)
%             if (distVect(j) > 1)
%                 continue;
%             end
%             if (sum(inliers==j) == 0)
%                 continue;
%             end
            tmp1(j,:)=corners1(C(j),1:2);
            tmp2(j,:)=corners2(j,1:2);
        end
        
        [tmp2 tmp1]=PreProcessCp2tform(tmp2,tmp1);
        if (length(tmp1) < 4)
            diffs = 0;
            nMask = 0;
        else
            t_poly = cp2tform(tmp2,tmp1,'piecewise linear');
            mask=ones(size(img1));
            tmpImg = imtransform(img1,t_poly,'FillValues',.3);
            tmpImg=imresize(tmpImg, size(img2));
            tmpImg=tmpImg-img2;
            diffs = sum(sum(tmpImg>50));
            mask = imtransform(mask,t_poly);
            mask=imresize(mask, size(img2));
            nMask = sum(sum(mask ~= 0));
        end
        
        if (stat)
            fid=fopen(statPath, 'a');
            fprintf(fid, '%s\t%f\t%f\t%f\t%s\t%d\t%s\t%d\t%d\t%d\n', char(currPath), dist(i), ang(i), param, method, nDim, opt.method, length(C), diffs, nMask);
            fclose(fid);
        end
        
        
        if (show)
            tmp1=ones(2,2);
            tmp2=ones(2,2);
            for j=1:length(C)
                if (distVect(j) > 1)
                    continue;
                end
                if (sum(inliers==j) == 0)
                    continue;
                end
                color=[rand, rand, rand];
                subplot(1,2,1);
                hold on;
                plot(corners1(C(j), 1), corners1(C(j), 2), 'k.','Color', color);
                subplot(1,2,2);
                hold on;
                plot(corners2(j, 1), corners2(j, 2), 'k.','Color', color);
                
                tmp1(j,:)=corners1(C(j),1:2);
                tmp2(j,:)=corners2(j,1:2);
            end
            
            figure(10);
            imshow(tmpImg);
            
            saveImgs(currPath, i);
            saveFileName = strcat(currPath, 'out/data', num2str(i), '.txt')            
            s1 = size(tmp1,1);
            s2 = size(tmp1,1);
            tmp1=tmp1';
            tmp2=tmp2';            
            dlmwrite(saveFileName, s1);
            dlmwrite(saveFileName, s2, '-append');
            dlmwrite(saveFileName, tmp1, '-append');
            dlmwrite(saveFileName, tmp2, '-append');
                        
%             k = waitforbuttonpress
%             if (k == 1)
%                 close all;
%                 return;
%             end                        
        end
    catch em
        fid=fopen(errPath, 'a');                
%         fprintf(fid, '%s\t%f\t%s\t%d\t%s\t%s\t%f\t%f\n', char(currPath), param, char(method), nDim, char(opt.method), em.identifier, i, i);
        fprintf(fid, '%s\t%s\t%s\n', char(currPath), method, getReport(em, 'extended'));
        fclose(fid);
    end    
end

t=1;