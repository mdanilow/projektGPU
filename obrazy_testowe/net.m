close all;
clear all;

size = 256;

img = zeros(size, size, 3);
%% siatka
for row = 1:2:size
   for col = 1:2:size
       
       img(row,col,:) = 255;
   end
end
figure();
imshow(img);
finalMat = img;
imwrite(finalMat,['siatka', num2str(size),'.bmp'],'bmp');