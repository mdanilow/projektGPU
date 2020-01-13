clear all
close all
kernel = parallel.gpu.CUDAKernel('kernel_one.ptx', 'kernel_one.cu');

im = imread('peppers.png');
im = rgb2gray(im);
im = uint8(imbinarize(im) .* 255);
figure(1);
imshow(im);

im_height = size(im,1);
im_width = size(im,2);

block_width = 16;
block_height = 16;
kernel.ThreadBlockSize = [block_width, block_height];
kernel.GridSize = [ceil(im_width/block_width), ceil(im_height/block_height)];

result = zeros(size(im));
result = feval(kernel, im, result, im_height, im_width);
result = gather(result);
figure(2);
imshow(result, []);