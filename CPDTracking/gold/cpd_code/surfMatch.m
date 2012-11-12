function [points1, points2] = surfMatch(I1, I2)      
  % Get the Key Points
    Options.upright=true;
    Options.tresh=0.0001;
    Ipts1=OpenSurf(I1,Options);
    Ipts2=OpenSurf(I2,Options);
  % Put the landmark descriptors in a matrix
    D1 = zeros([64 length(Ipts1)]);
    D2 = zeros([64 length(Ipts2)]);
    for i=1:length(Ipts1), D1(:,i)=Ipts1(i).descriptor; end
    for i=1:length(Ipts2), D2(:,i)=Ipts2(i).descriptor; end
    
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
                points1(count, :) = [ Ipts1(best2(i)).x Ipts1(best2(i)).y ];
                points2(count, :) = [ Ipts2(i).x Ipts2(i).y ];
                count = count + 1;
            end
        end
    end
    
    points1 = points1(1:(count - 1), :);
    points2 = points2(1:(count - 1), :);
end