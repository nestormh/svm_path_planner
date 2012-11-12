path='/home/neztol/doctorado/Datos/EstadisticasITER/ManzanillaTripodeLowRes/';
filepath='datos.txt';
ext='.JPG';
protocol='file://';
paramFAST=10;
paramCorner=0.2; %0.2;
method='surf'
% method=num2str(paramFAST)

filepath=[path,filepath];

[dist, ang, imgName] = textread(filepath, '%f %f %s', 'headerlines', 2);

fid = fopen(filepath);

line = [path,fgetl(fid),ext];

fclose(fid);

img1=imread(line);
img1=rgb2gray(img1);

descPath = [line, '.surfDescriptors.txt'];
detectPath = [line, '.surfDetectors.txt'];

[x1, y1, scale1, strength1, ori1, laplace1, time1] = textread(detectPath, '%f %f %f %f %f %d %d');

format = '';
for i=1:63
    format = [format,'%f '];
end
format = [format,'%f'];

[ desc1, desc2, desc3, desc4, desc5, desc6, desc7, desc8, desc9, desc10, desc11, desc12, desc13, desc14, desc15, desc16, desc17, desc18, desc19, desc20, desc21, desc22, desc23, desc24, desc25, desc26, desc27, desc28, desc29, desc30, desc31, desc32, desc33, desc34, desc35, desc36, desc37, desc38, desc39, desc40, desc41, desc42, desc43, desc44, desc45, desc46, desc47, desc48, desc49, desc50, desc51, desc52, desc53, desc54, desc55, desc56, desc57, desc58, desc59, desc60, desc61, desc62, desc63, desc64 ] = textread(descPath, format);
img1Desc = [ desc1, desc2, desc3, desc4, desc5, desc6, desc7, desc8, desc9, desc10, desc11, desc12, desc13, desc14, desc15, desc16, desc17, desc18, desc19, desc20, desc21, desc22, desc23, desc24, desc25, desc26, desc27, desc28, desc29, desc30, desc31, desc32, desc33, desc34, desc35, desc36, desc37, desc38, desc39, desc40, desc41, desc42, desc43, desc44, desc45, desc46, desc47, desc48, desc49, desc50, desc51, desc52, desc53, desc54, desc55, desc56, desc57, desc58, desc59, desc60, desc61, desc62, desc63, desc64 ]
clear desc*;

corners1 = [ x1, y1 ];

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
    img2=rgb2gray(img2);
    
    descPath = [imName, '.surfDescriptors.txt'];
    detectPath = [imName, '.surfDetectors.txt'];

    [x2, y2, scale2, strength2, ori2, laplace2, time2] = textread(detectPath, '%f %f %f %f %f %d %d');

    [ desc1, desc2, desc3, desc4, desc5, desc6, desc7, desc8, desc9, desc10, desc11, desc12, desc13, desc14, desc15, desc16, desc17, desc18, desc19, desc20, desc21, desc22, desc23, desc24, desc25, desc26, desc27, desc28, desc29, desc30, desc31, desc32, desc33, desc34, desc35, desc36, desc37, desc38, desc39, desc40, desc41, desc42, desc43, desc44, desc45, desc46, desc47, desc48, desc49, desc50, desc51, desc52, desc53, desc54, desc55, desc56, desc57, desc58, desc59, desc60, desc61, desc62, desc63, desc64 ] = textread(descPath, format);
    img2Desc = [ desc1, desc2, desc3, desc4, desc5, desc6, desc7, desc8, desc9, desc10, desc11, desc12, desc13, desc14, desc15, desc16, desc17, desc18, desc19, desc20, desc21, desc22, desc23, desc24, desc25, desc26, desc27, desc28, desc29, desc30, desc31, desc32, desc33, desc34, desc35, desc36, desc37, desc38, desc39, desc40, desc41, desc42, desc43, desc44, desc45, desc46, desc47, desc48, desc49, desc50, desc51, desc52, desc53, desc54, desc55, desc56, desc57, desc58, desc59, desc60, desc61, desc62, desc63, desc64 ]
    clear desc*;

    corners2 = [ x2, y2 ];
   
    figure(4);
    subplot(1,2,2);
    image(img2/4);
    axis image;
    colormap(gray);
    hold on;
    plot(corners2(:,1), corners2(:,2), 'r.');
    hold off;    
    
    
    figure(1);
    opt.method='nonrigid_lowrank';
%     opt.lambda = 10;
    opt.corresp=1;
%     opt.normalize=0;
%     opt.viz=1;
%     opt.tol=1e-10
    [Transform, C]=cpd_register(img1Desc,img2Desc, opt);

    figure(5);
    subplot(1,2,1);
    image(img1/4);
    axis image;
    colormap(gray);

    subplot(1,2,2);
    image(img2/4);
    axis image;
    colormap(gray);

%     matName=strcat(imgName(i), '.', num2str(paramFAST), '.mat')
%     matName=strcat(imgName(i), '.', 'canny', '.mat')
    
%     save(char(matName),'corners1','corners2','distVect','C','Transform');
    
    x1 = [corners1(C(:),1), corners1(C(:),2), ones(length(C),1)]';
    x2 = [corners2(:,1), corners2(:,2), ones(length(C),1)]';
    [F, inliers] = ransacfitfundmatrix(x1, x2, 0.001);

    for j=1:length(C)

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

beep