fid=fopen('/tmp/CPD/outputCPD.txt', 'r');

for i=1:12
    fgetl(fid);
end
nDims = fscanf(fid, 'nDims = %d\n');
distThresh = fscanf(fid, 'distThresh = %lf\n');
for i=15:18
    fgetl(fid);
end

xSize = str2double(fgetl(fid));
X = cell2mat(textscan(fid, '%f\t%f\t%f\t%f\t%f\t%f\n', xSize));
ySize = str2double(fgetl(fid));
Y = cell2mat(textscan(fid, '%f\t%f\t%f\t%f\t%f\t%f\n', ySize));
yTransfSize = str2double(fgetl(fid));
YTransf = cell2mat(textscan(fid, '%f\t%f\t%f\t%f\t%f\t%f\n', yTransfSize));
CSize = str2double(fgetl(fid));
C = cell2mat(textscan(fid, '%d\n', CSize));

fclose(fid);

figure(1);
hold on
for i=1:ySize
    p0 = Y(i,:);
    p1 = YTransf(i,:);
    p2 = X(C(i), :);

    if (pdist([p1(1:nDims);p2(1:3)], 'euclidean') < distThresh)
        plot3([p0(1) p2(1)], [p0(2) p2(2)], [p0(3) p2(3)], 'r-');
    end
%     break;
end
axis([0 20 0 20 0 20]);
hold off

figure(2);
hold on
for i=1:ySize
    p0 = Y(i,:);
    p1 = YTransf(i,:);
    p2 = X(C(i), :);

    if (pdist([p1(1:nDims);p2(1:3)], 'euclidean') < distThresh)
        plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], 'r-');
    end
%     break;
end
axis([0 20 0 20 0 20]);
hold off