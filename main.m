clear all
close all
kernel_1 = parallel.gpu.CUDAKernel('kernel_one.ptx', 'kernel_one.cu');
kernel_2 = parallel.gpu.CUDAKernel('kernel_two.ptx', 'kernel_two.cu');

im = imread('peppers.png');
im = rgb2gray(im);
im = uint8(imbinarize(im) .* 255);
imshow(im);

im_height = size(im,1);
im_width = size(im,2);
% dummy_img = zeros(im_height, im_width);

% for row = 1:im_height

%     for col = 1:im_width
        
%         dummy_img(row, col) = row-1 + (col-1)*im_height;
%     end
% end

block_width = 16;
block_height = 16;
kernel_1.ThreadBlockSize = [block_width, block_height];
kernel_1.GridSize = [ceil(im_width/block_width), ceil(im_height/block_height)];

result = zeros(size(im));
result = feval(kernel_1, im, result, im_height, im_width);
result = gather(result);

%result = feval(kernel_2, im, result, im_height, im_width);
%result = gather(result);
