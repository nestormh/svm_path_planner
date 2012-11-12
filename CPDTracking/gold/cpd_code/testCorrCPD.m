function t = testCorrCPD(currPath, filepath, ext, show, method, stat, statPath, errPath)
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

[dist, ang, imgName] = textread(filepath, '%f %f %s', 'headerlines', 2);
fid = fopen(filepath);
line = [currPath,fgetl(fid),ext];
fclose(fid);
img1=imread(line);
img1=rgb2gray(img1);
       
imgName(:)=strcat(currPath,imgName(:),ext);

% Recorremos todas las imágenes del fichero
for i=1:length(imgName)      
    clc;
    try
%         if (abs(dist(i)) > 1)
%             continue;
%         end
%         if ((dist(i) == -1) && (ang(i) < 0))
%             continue;
%         end
%         if ((dist(i) == 1) && (ang(i) > 0))
%             continue;
%         end
%         if (abs(ang(i)) > 10)
%             continue;
%         end
        
        imName = char(imgName(i));
        img2=imread(imName);
        img2=rgb2gray(img2);
        
        % Mostramos la evolución del CPD
        try
            tstart = tic;
            [corners1,corners2]=corrMatch(img1,img2,40,3);
            telapsed = toc(tstart);
        catch me
            fid=fopen(errPath, 'a');                
            fprintf(fid, '%s\t%f\t%f\t%s\t%s\n', char(currPath), dist(i), ang(i), method, getReport(me, 'extended'));
            fclose(fid);            
            continue;
        end    
                
        if (show)
            I = zeros([size(img1,1) size(img1,2)*2 size(img1,3)]);
            I(:,1:size(img1,2),:)=img1; I(:,size(img1,2)+1:size(img1,2)+size(img2,2),:)=img2;
            figure(5), imshow(I/255); hold on;   
            for pts=1:length(corners1),
                c=rand(1,3);        
                plot([corners1(pts,1) corners2(pts,1) + size(img1,2)],[corners1(pts,2) corners2(pts,2) ],'o','Color',c)
            end
        end
        
        % Con un RANSAC terminamos de eliminar outliers
        x1 = [corners1(:,1), corners1(:,2), ones(length(corners1),1)]';
        x2 = [corners2(:,1), corners2(:,2), ones(length(corners1),1)]';
        [F, inliers] = ransacfitfundmatrix(x1, x2, 0.001);
        
        % Obtenemos la diferencia de imágenes, usando los puntos seleccionados
        tmp1=ones(2,2);
        tmp2=ones(2,2);
        for j=1:length(corners1)
            if (sum(inliers==j) == 0)
                continue;
            end
            tmp1(j,:)=corners1(j,1:2);
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
%             fprintf(fid, '%s\t%f\t%f\t%s\t%d\t%d\t%d\t%f\n', char(currPath), dist(i), ang(i), method, length(corners1), diffs, nMask, telapsed);
            fprintf(fid, '%s\t%f\t%f\t%f\t%s\t%d\t%s\t%d\t%d\t%d\t%f\n', char(currPath), dist(i), ang(i), 0, method, 0, method, length(corners1), diffs, nMask, telapsed);
            printf('Hola\n');
            printf('%s\t%f\t%f\t%f\t%s\n', char(currPath), dist(i), ang(i), 0, method);
            printf('%s\t%f\t%f\t%f\t%s\t%d\t%s\t%d\t%d\t%d\t%f\n', char(currPath), dist(i), ang(i), 0, method, 0, method, length(corners1), diffs, nMask, telapsed);
            fclose(fid);
        end
        
        
        if (show)
            corners1=tmp1;
            corners2=tmp2;

            I = zeros([size(img1,1) size(img1,2)*2 size(img1,3)]);
            I(:,1:size(img1,2),:)=img1; I(:,size(img1,2)+1:size(img1,2)+size(img2,2),:)=img2;
            figure(10), imshow(I/255); hold on;   
            for pts=1:length(corners1),
                c=rand(1,3);        
                plot([corners1(pts,1) corners2(pts,1) + size(img1,2)],[corners1(pts,2) corners2(pts,2) ],'o','Color',c)
            end
            
            k = waitforbuttonpress
            if (k == 1)
                close all;
                return;
            end
        end
    catch em
        fid=fopen(errPath, 'a');                
%         fprintf(fid, '%s\t%s\t%s\t%f\t%f\n', char(currPath), char(method), nDim, char(opt.method), em.identifier, dist(i), ang(i));
        fprintf(fid, '%s\t%f\t%f\t%s\t%s\n', char(currPath), dist(i), ang(i), method, getReport(em, 'extended'));
        fclose(fid);
    end    
end

t=1;