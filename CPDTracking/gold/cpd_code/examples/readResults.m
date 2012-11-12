clear;

path='/home/neztol/doctorado/Datos/EstadisticasITER/ManzanillaTripodeLowRes/';
filepath='datos.txt';
ext='.JPG';
protocol='file://';
method='canny2DSparse5' % '10', 'canny', 'canny3D', 'canny3DSparse5', canny2DSparse5
methods=[{'10'}, {'canny'}, {'canny3D'}, {'canny2DSparse5'}, {'canny3DSparse5'}];
pasos=[1,10,10,1,1];
paso = 5;
% method=num2str(paramFAST)

filepath=[path,filepath];

[dist, ang, imgName] = textread(filepath, '%f %f %s', 'headerlines', 2);

fid = fopen(filepath);

line = [path,fgetl(fid),ext];

fclose(fid);

img1=imread(line);
% img1 = imresize(img1, [240 320]);
img1 = double(img1(:,:,2));    
    
imgName(:)=strcat(path,imgName(:),ext);

for i=1:length(imgName)      
    
    if (abs(dist(i)) > 1) 
        continue;
    end
    if ((dist(i) == -1) && (ang(i) < 0))
        continue;        
    end
    if ((dist(i) == 1) && (ang(i) > 0))
        continue;
    end
    if (abs(ang(i)) > 5)
        continue;
    end
%     if (abs(ang(i)) > 0)
%         continue;
%     end
%     if (abs(dist(i)) > 0) 
%         continue;
%     end
    
    for m=1:length(methods)

        method = methods(m);
        paso = pasos(m);
        
	imName = char(imgName(i));
    img2=imread(imName);
%     img2 = imresize(img2, [240 320]);
    img2 = double(img2(:,:,2));    
    
%     figure('Name', char(method),'NumberTitle','off');
    figure('Name', char(method));
    subplot(1,2,1);
    image(img1/4);
    axis image;
    colormap(gray);

    subplot(1,2,2);
    image(img2/4);
    axis image;
    colormap(gray);

    matName=char(strcat(imgName(i), '.', method,'.mat'));

    clear corners1 corners2 C distVect Transform;
    
    load(matName);
    
%     distVect=sqrt((corners1(C(:), 1) - Transform.Y(:, 1)).^2 + (corners1(C(:), 2) - Transform.Y(:, 2)).^2);
    
    x1 = [corners1(C(:),1), corners1(C(:),2), ones(length(C),1)]';
    x2 = [corners2(:,1), corners2(:,2), ones(length(C),1)]';
    [F, inliers] = ransacfitfundmatrix(x1, x2, 0.001);    
    
    wSize = 3;
    thresh = 0.4;
    valid1 = ones(1,1);
    valid2 = ones(1,1);
    pos = 1;
    for j=1:paso:length(C)
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
        valid1(pos, 2) = 0;
        valid1(pos, :) = corners1(C(j), :);
        valid2(pos, 2) = 0;
        valid2(pos, :) = corners2(j,:);
        pos = pos + 1;
        
        
%         x1a = corners1(C(j), 1) - wSize;
%         x1b = corners2(j, 1) - wSize;
%         if (x1a < 1)
%             x1b = x1b + abs(x1a) + 1;
%             x1a = 1;
%         end
%         if (x1b < 1)
%             x1a = x1a + abs(x1b) + 1;
%             x1b = 1;
%         end
%         y1a = corners1(C(j), 2) - wSize;
%         y1b = corners2(j, 2) - wSize;
%         if (y1a < 1)
%             y1b = y1b + abs(y1a) + 1;
%             y1a = 1;
%         end
%         if (y1b < 1)
%             y1a = y1a + abs(y1b) + 1;
%             y1b = 1;
%         end
%         x2a = corners1(C(j), 1) + wSize;
%         x2b = corners2(j, 1) + wSize;
%         if (x2a > size(img1,2))
%             x2b = x2b + (x2a - size(img1, 2));
%             x2a = size(img1, 2);
%         end
%         if (x2b > size(img2,2))
%             x2a = x2a + (x2b - size(img2, 2));
%             x2b = size(img2, 2);
%         end
%         y2a = corners1(C(j), 2) + wSize;
%         y2b = corners2(j, 2) + wSize;
%         if (y2a > size(img1,1))
%             y2b = y2b + (y2a - size(img1, 1));
%             y2a = size(img1, 1);
%         end
%         if (y2b > size(img2,1))
%             y2a = y2a + (y2b - size(img2, 1));
%             y2b = size(img2, 1);
%         end
%         
%         area1 = img1(y1a:y2a, x1a:x2a);
%         area2 = img2(y1b:y2b, x1b:x2b);   
%         corr = corrcoef(area1(:), area2(:));
%         corr = corr(1,2);
%         
%         color=[rand, rand, rand];
%         subplot(1,2,1);
%         hold on;
%         if (corr < thresh)
% %             plot(corners1(C(j), 1), corners1(C(j), 2), 'kx','Color', color);
%         else
%             plot(corners1(C(j), 1), corners1(C(j), 2), 'k.','Color', color);
%         end
%         subplot(1,2,2);
%         hold on;   
%         if (corr < thresh)
% %             plot(corners2(j, 1), corners2(j, 2), 'kx','Color', color);
%         else
%             plot(corners2(j, 1), corners2(j, 2), 'k.','Color', color);
%         end
    end    
    
%     [valid1, valid2] = cleanDelaunay(img1, img2, valid1, valid2);
%     
%     x1 = [valid1(:,1), valid1(:,2), ones(length(valid1),1)]';
%     x2 = [valid2(:,1), valid2(:,2), ones(length(valid2),1)]';
%     [F, inliers] = ransacfitfundmatrix(x1, x2, 0.001);    
%     
%     figure(10);
%     subplot(1,2,1);
%     image(img1/4);
%     axis image;
%     colormap(gray);
% 
%     subplot(1,2,2);
%     image(img2/4);
%     axis image;
%     colormap(gray);
%     
%     for j=1:length(valid1)
%         if (sum(inliers==j) == 0)
%             continue;
%         end
%         color=[rand, rand, rand];
%         subplot(1,2,1);
%         hold on;
%         plot(valid1(j, 1), valid1(j, 2), 'k.','Color', color);
%         subplot(1,2,2);
%         hold on;
%         plot(valid2(j, 1), valid2(j, 2), 'k.','Color', color);
%     end
    
    end
    
    k = waitforbuttonpress
    if (k == 1)
        break;
    end
    
    close all;
%     break;
end
% x1 = [corners1(C(:),1), corners1(C(:),2), ones(length(C),1)]';
% x2 = [corners2(:,1), corners2(:,2), ones(length(C),1)]';
% [F, inliers] = ransacfitfundmatrix(x1, x2, 0.001);
% for i=1:length(imgName)
%     corners1=zeros(size(pairs,3),2);
%     corners2=zeros(size(pairs,3),2);
%     distVect=zeros(size(pairs,3),1);
%     corners1(:,:)=pairs(i,1,:,:);
%     corners2(:,:)=pairs(i,2,:,:);
%     distVect(:)=pairs(i,3,:,1);
%     for j=1:length(C)
%         if (distVect(j) > 1)
%             continue;
%         end
%         color=[rand, rand, rand];
%         subplot(1,2,1);
%         hold on;
%         plot(corners1(C(j), 1), corners1(C(j), 2), 'k.','Color', color);
%         subplot(1,2,2);
%         hold on;   
%         plot(corners2(j, 1), corners2(j, 2), 'k.','Color', color);
%     end
% 
%     
%     break;
% end