function [ line ] = HoughDetect( vDisparityImage )
%PRUEBAHOUGH Summary of this function goes here
%   Detailed explanation goes here
     
    close all

    I  = vDisparityImage;
%       rotI = imrotate(I,33,'crop');
    rotI = I;
    BW = I;
      
       figure
       imshow(BW)
       
       [H,T,R] = hough(BW);
             
       figure
       imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');
       xlabel('\theta'), ylabel('\rho');
       axis on, axis normal, hold on;

       % Limitar la búsqueda
       inf = find(T > -50, 1);
       sup = find(T > -10, 1);
       H1 = H;
       H1(:, 1:inf) = 0;
       sizeH = size(H1);
       H1(:, sup:sizeH(2)) = 0;

       P  = houghpeaks(H1,12,'threshold',ceil(0.1*max(H(:))));                
       
       x = T(P(:,2)); y = R(P(:,1));
       plot(x,y,'s','color','white');
       
       % Find lines and plot them
       lines = houghlines(BW,T,R,P,'FillGap',20,'MinLength',100);
             
       figure, imshow(rotI), hold on
       max_len = 0;
       for k = 1:length(lines)
         xy = [lines(k).point1; lines(k).point2];
         plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','green');
 
         % plot beginnings and ends of lin
%         plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%         plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
 
         % determine the endpoints of the longest line segment 
         lines(k).len = norm(lines(k).point1 - lines(k).point2);
         
         for i=1:k                              % Ordenar las líneas por longitud de mayor a menor
            if ( lines(i).len < lines(k).len)
                aux = lines(k);
                lines(k)=lines(i);
                lines(i)=aux;
            end
         end
       end


       i = 1;
       while ((lines(i).rho/sin(lines(i).theta*pi/180) < 0) && (i < length(lines)))
           i = i + 1;
       end

       i
       
       xy = [lines(i).point1; lines(i).point2];
       plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','cyan');
       
       line = lines(i);