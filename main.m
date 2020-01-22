clear all
close all

TILE_SIZE = 16;
TILES_PER_BLOCK_KER2 = 4;

kernel_1 = parallel.gpu.CUDAKernel('kernel_one.ptx', 'kernel_one.cu');
kernel_2 = parallel.gpu.CUDAKernel('kernel_two_ja.ptx', 'kernel_two_ja.cu');

im = imread('coins.png');
im = imresize(im, [240, 304]);
im = imfill(im);

% im = imread('obrazy_testowe/siatka256.bmp')
% im = rgb2gray(im);

im = int32(imbinarize(im) .* 255);
imshow(im, []);

im_height = size(im,1);
im_width = size(im,2);

% dummy_img = zeros(im_height, im_width);

% for row = 1:im_height

%     for col = 1:im_width
        
%         dummy_img(row, col) = row-1 + (col-1)*im_height;
%     end
% end

kernel_1.ThreadBlockSize = [TILE_SIZE, TILE_SIZE];
kernel_1.GridSize = [ceil(im_width/TILE_SIZE), ceil(im_height/TILE_SIZE)];

kernel_2.ThreadBlockSize = [TILES_PER_BLOCK_KER2, TILES_PER_BLOCK_KER2, TILE_SIZE];
kernel_2.GridSize = [ceil(im_width/(TILE_SIZE*TILES_PER_BLOCK_KER2)), ceil(im_height/(TILE_SIZE*TILES_PER_BLOCK_KER2))];

ker1_result = zeros(size(im));
ker1_result = feval(kernel_1, im, ker1_result, im_height, im_width);
ker1_result = gather(ker1_result);
ker1_result_copy = ker1_result;
figure();
imshow(ker1_result, []);
figure();
subplot(1, 2, 1);
imshow(im, []);
subplot(1, 2, 2);
imshow(ker1_result_copy, []);

ker2_result = zeros(size(im));
ker2_result = feval(kernel_2, im, ker1_result, im_height, im_width);
ker2_result = gather(ker2_result);
