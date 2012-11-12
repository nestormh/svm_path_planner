nTest=0;
if (nTest==0) 
    path='/home/neztol/doctorado/Datos/DB/Rutas/pruebaITERBase2/';
    ext='.png';
    maxIt=601;
end
paramFAST=30;
paramCorner=0.2;

for i=0:maxIt      
    imName = strcat(path, 'Camera', '0/Image', num2str(i), ext);
    img1 = imread(imName);
    img1 = imresize(img1, [240 320]);
    img1 = imcrop(img1,[10 0 320 240]);
%     img1 = double(img1(:,:,2));    
    img1=rgb2gray(img1);
    
    corners1 = fast_corner_detect_9(img1, paramFAST);
% %     corners1 = fast_nonmax(img1, paramFAST, corners1);
%     cout = corner(img1, [], [], [], paramCorner);
%     corners1 = [cout(:,2), cout(:,1)];
        
    imName = strcat(path, 'Camera', '1/Image', num2str(i), ext);
    img2 = imread(imName);
    img2 = imresize(img2, [240 320]);
    img2 = imcrop(img2,[10 0 320 240]);
%     img2 = double(img2(:,:,2));   
    img2=rgb2gray(img2);
    
    corners2 = fast_corner_detect_9(img2, paramFAST);
%     corners2 = fast_nonmax(img2, paramFAST, corners2);
%     cout=corner(img2, [], [], [], paramCorner);
%     corners2=[cout(:,2), cout(:,1)];

    figure(1);
    subplot(1,2,1);
    image(img1/4);
    axis image;
    colormap(gray);
    hold on;
    plot(corners1(:,1), corners1(:,2), 'r.');
    hold off;
    subplot(1,2,2);
    image(img2/4);
    axis image;
    colormap(gray);
    hold on;
    plot(corners2(:,1), corners2(:,2), 'r.');
    hold off;
    
    corners1 = [corners1(:,1), corners1(:, 2), corners1(:,2)];
    corners2 = [corners2(:,1), corners2(:, 2), corners2(:,2)];
    for j=1:length(corners1)
        corners1(j,:)=[corners1(j,1), corners1(j,2), img1(corners1(j,2), corners1(j,1))];
    end
    for j=1:length(corners2)
        corners2(j,:)=[corners2(j,1), corners2(j,2), img2(corners2(j,2), corners2(j,1))];
    end
    
    figure(2);
    opt.method='nonrigid';
    opt.corresp=1;
%     opt.normalize=0;
%     opt.viz=1;
%     opt.tol=1e-10
    [Transform, C]=cpd_register(corners1,corners2, opt);

    figure(3);
    subplot(1,2,1);
    image(img1/4);
    axis image;
    colormap(gray);

    subplot(1,2,2);
    image(img2/4);
    axis image;
    colormap(gray);

    distVect=sqrt((corners1(C(:), 1) - Transform.Y(:, 1)).^2 + (corners1(C(:), 2) - Transform.Y(:, 2)).^2);
    
    for j=1:length(C)
        if (distVect(j) > 5)
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

