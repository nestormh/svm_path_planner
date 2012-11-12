function [points1, points2] = corrMatch(I1, I2, param, win)      
    corners1 = fast_corner_detect_9(I1, param);
    corners1 = fast_nonmax(double(I1), param, corners1);
    [r,c,v]=find(corners1(:,1)>win);
    corners1=corners1(r,:);
    [r,c,v]=find(corners1(:,2)>win);
    corners1=corners1(r,:);
    [r,c,v]=find(corners1(:,1)<(size(I1,2) - (win - 1)));
    corners1=corners1(r,:);
    [r,c,v]=find(corners1(:,2)<(size(I1,1) - (win - 1)));
    corners1=corners1(r,:);
    
    corners2 = fast_corner_detect_9(I2, param);
    corners2 = fast_nonmax(double(I2), param, corners2);
    [r,c,v]=find(corners2(:,1)>win);
    corners2=corners2(r,:);
    [r,c,v]=find(corners2(:,2)>win);
    corners2=corners2(r,:);
    [r,c,v]=find(corners2(:,1)<(size(I2,2) - (win - 1)));
    corners2=corners2(r,:);
    [r,c,v]=find(corners2(:,2)<(size(I2,1) - (win - 1)));
    corners2=corners2(r,:);    
    
    descSize=(2 * win + 1);    
    D1 = zeros([descSize*descSize length(corners1)]);
    D2 = zeros([descSize*descSize length(corners2)]);       
    
    for i = 1:length(corners1)
        iniX=corners1(i,1);
        iniY=corners1(i,2);
        c=1;
        for j=-win:win
            for k=-win:win
                D1(c,i) = I1(iniY + k, iniX + j);
                c = c+1;
            end
        end
    end
       
    for i = 1:length(corners2)
        iniX=corners2(i,1);
        iniY=corners2(i,2);
        c=1;
        for j=-win:win
            for k=-win:win
                D2(c,i) = I2(iniY + k, iniX + j);
                c = c+1;
            end
        end
    end
    
       corr=zeros(size(D1,2), size(D2,2));   
   
    for i=1:size(D1,2)
        for j=1:size(D2,2)
            r=corrcoef(D1(:, i)', D2(:, j)');
            corr(i,j)=r(1,2);
        end
    end
    [max1, best1]=max(corr, [], 2);
    [max2, best2]=max(corr);
    
    points1 = zeros(size(D1,2),2);
    points2 = zeros(size(D2,2),2);
    count=1;
    for i=1:length(best2)                
        if (best1(best2(i)) == i)
            if (max2(i) > 0.95)
                points1(count, :) = [ corners1(best2(i),1) corners1(best2(i),2) ];
                points2(count, :) = [ corners2(i,1) corners2(i,2) ];
                count = count + 1;
            end
        end
    end
    
    points1 = points1(1:(count - 1), :);
    points2 = points2(1:(count - 1), :);

    I = zeros([size(I1,1) size(I1,2)*2 size(I1,3)]);
    I(:,1:size(I1,2),:)=I1; I(:,size(I1,2)+1:size(I1,2)+size(I2,2),:)=I2;
    figure(10), imshow(I/255); hold on;
    for pts=1:length(corners1),
        c=rand(1,3);
        plot([corners1(pts,1) corners2(pts,1) + size(I1,2)],[corners1(pts,2) corners2(pts,2) ],'o','Color',c)
    end
end
