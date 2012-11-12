path='/home/neztol/doctorado/Datos/EstadisticasITER/ManzanillaTripodeLowRes/';
filepath='datos.txt';
ext='.JPG';
protocol='file://';
paramFAST=10;
paramCorner=0.2; %0.2;
paso = 5;
method='canny2DSparse5'
% method=num2str(paramFAST)

filepath=[path,filepath];

[dist, ang, imgName] = textread(filepath, '%f %f %s', 'headerlines', 2);

fid = fopen(filepath);

line = [path,fgetl(fid),ext];

fclose(fid);

img1=imread(line);
% img1 = imresize(img1, [240 320]);
img1 = double(img1(:,:,2));    
% img1=rgb2gray(img1);
    
borders1=edge(img1, 'canny');

corners1 = ones(1,2);
index=1;
pos = 0;
for a=1:size(borders1,1)
    for b=1:size(borders1,2)
        if (borders1(a,b)==1)
            pos = pos + 1;
            if (mod(pos, paso) ~= 0) 
                continue;            
            end
            corners1(index,1) = b;
            corners1(index,2) = a;
            index = index + 1;
        end
    end
end

% corners1 = fast_corner_detect_9(img1, paramFAST);
% corners1 = fast_nonmax(img1, paramFAST, corners1);
% cout=corner(img1, [], [], [], paramCorner);
% corners1=[cout(:,2), cout(:,1)];
    
figure(4);
subplot(1,2,1);
image(img1/4);
axis image;
colormap(gray);
hold on;
plot(corners1(:,1), corners1(:,2), 'r.');
hold off;

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
    if (abs(ang(i)) > 10)
        continue;
    end
%     if (abs(ang(i)) > 0)
%         continue;
%     end
%     if (abs(dist(i)) > 0) 
%         continue;
%     end
    
	imName = char(imgName(i));
    img2=imread(imName);
%     img2 = imresize(img2, [240 320]);
    img2 = double(img2(:,:,2));   
%     img2=rgb2gray(img2);
    
%     corners2 = fast_corner_detect_9(img2, paramFAST);
%     corners2 = fast_nonmax(img2, paramFAST, corners2);
%     cout=corner(img2, [], [], [], paramCorner);
%     corners2=[cout(:,2), cout(:,1)];
    borders2=edge(img2, 'canny');

    corners2 = ones(1,2);
    index=1;
    pos = 0;
    for a=1:size(borders2,1)
        for b=1:size(borders2,2)            
            if (borders2(a,b)==1)
                pos = pos + 1;
                if (mod(pos, paso) ~= 0) 
                    continue;
                end
                corners2(index,1) = b;
                corners2(index,2) = a;
                index = index + 1;
            end
        end
    end


    figure(4);
    subplot(1,2,2);
    image(img2/4);
    axis image;
    colormap(gray);
    hold on;
    plot(corners2(:,1), corners2(:,2), 'r.');
    hold off;    
    
%     corners1 = [corners1(:,1), corners1(:, 2), corners1(:,2)];
%     corners2 = [corners2(:,1), corners2(:, 2), corners2(:,2)];
% %     filt = medfilt2(img1);
%     for j=1:length(corners1)
%         corners1(j,:)=[corners1(j,1), corners1(j,2), img1(corners1(j,2), corners1(j,1))];
%     end
% %     filt = medfilt2(img1);
%     for j=1:length(corners2)
%         corners2(j,:)=[corners2(j,1), corners2(j,2), img2(corners2(j,2), corners2(j,1))];
%     end
    
    figure(1);
    corners1 = corners1(:,1:2);
    corners2 = corners2(:,1:2);
    
    figure(1);
    opt.method='nonrigid_lowrank';
%     opt.lambda = 10;
    opt.corresp=1;
%     opt.max_it=300;
%     opt.normalize=0;
%     opt.viz=1;
%     opt.tol=1e-10
    [Transform, C]=cpd_register(corners1,corners2, opt);

    figure(5);
    subplot(1,2,1);
    image(img1/4);
    axis image;
    colormap(gray);

    subplot(1,2,2);
    image(img2/4);
    axis image;
    colormap(gray);

    distVect=sqrt((corners1(C(:), 1) - Transform.Y(:, 1)).^2 + (corners1(C(:), 2) - Transform.Y(:, 2)).^2);

%     matName=strcat(imgName(i), '.', num2str(paramFAST), '.mat')
    matName=strcat(imgName(i), '.', method, '.mat')
    
%     save(char(matName),'corners1','corners2','distVect','C','Transform');
    
    x1 = [corners1(C(:),1), corners1(C(:),2), ones(length(C),1)]';
    x2 = [corners2(:,1), corners2(:,2), ones(length(C),1)]';
    [F, inliers] = ransacfitfundmatrix(x1, x2, 0.001);

    for j=1:length(C)
%         if (distVect(j) > 1)
%             continue;
%         end
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
    end    
    break;
end

% close all;

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