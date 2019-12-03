% I = imread('slimak256.bmp');
% I = imread('siatka256.bmp');
I = imread('schody256.bmp');
gray = rgb2gray(I);
L = bwlabel(gray);

figure();
subplot(1,2,1);
imshow(I);
subplot(1,2,2);
imshow(L);
title(['number of labels:', num2str(max(max(L)))]);