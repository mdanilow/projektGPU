clear all
close all
kernel = parallel.gpu.CUDAKernel('kernel_one.ptx', 'kernel_one.cu');

im = imread('peppers.png');
im = rgb2gray(im);

im_height = size(im,1);
im_width = size(im,2);
dummy_img = zeros(im_height, im_width);

for row = 1:im_height

    for col = 1:im_width
        
        dummy_img(row, col) = row-1 + (col-1)*im_height;
    end
end

block_width = 16;
block_height = 16;
kernel.ThreadBlockSize = [block_width, block_height];
kernel.GridSize = [ceil(im_width/block_width), ceil(im_height/block_height)];

result = zeros(size(im));
result = feval(kernel, im, result, im_height, im_width);
result = gather(result);