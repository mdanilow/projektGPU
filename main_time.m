clear all
close all

%% initialization

TILE_SIZE = 16;
TILES_PER_BLOCK_KER2 = 4;

kernel_1 = parallel.gpu.CUDAKernel('kernel_one.ptx', 'kernel_one.cu');
kernel_2 = parallel.gpu.CUDAKernel('kernel_two_ja.ptx', 'kernel_two_ja.cu');

kernel_4 = parallel.gpu.CUDAKernel('kernel_four.ptx', 'kernel_four.cu');

kernel_1_iter = 1;
kernel_2_iter = 1;
kernel_4_iter = 1;
kernel_1_time = 0;
kernel_2_time = 0;
kernel_4_time = 0;

time_measure = false;% Set to false to show results. True for time measurements
n_max = 1100;       % Numer of iterations for time measurements


%im = imread('coins.png');
im = imread('schody256.bmp');

im = rgb2gray(im);
im = imfill(im);
im = int32(imbinarize(im) .* 255);
[im, levels] = preprocessImage(im);

if time_measure == false
    n_max = 1;
    figure();
    imshow(im, []);
    title('input image');
end

im_height = size(im,1);
im_width = size(im,2);


for n = drange(1:n_max)
    
    %% first kernel - local solution

    kernel_1.ThreadBlockSize = [TILE_SIZE, TILE_SIZE];
    kernel_1.GridSize = [ceil(im_width/TILE_SIZE), ceil(im_height/TILE_SIZE)];
    ker1_result = zeros(size(im));

    tic;
    ker1_result = feval(kernel_1, im, ker1_result, im_height, im_width);
    ker1_result = gather(ker1_result);
    if n > n_max/10
        kernel_1_time = kernel_1_time+(toc-kernel_1_time)/kernel_1_iter;
        kernel_1_iter = kernel_1_iter+1;
    end

    if time_measure == false
        figure();
        imshow(ker1_result, []);
        title('kernel 1 result');
    end
    %% second/third kernel loop - merging routine

    ker2_result = ker1_result;
    LEVEL = 0;
    ker2Grid = [ceil(im_width/(TILE_SIZE*TILES_PER_BLOCK_KER2)), ceil(im_height/(TILE_SIZE*TILES_PER_BLOCK_KER2))];
    ker2Outputs = zeros([im_height, im_width, 4], 'int32');

    i = 1;
    while min(ker2Grid) >= 2

        kernel_2.ThreadBlockSize = [TILES_PER_BLOCK_KER2, TILES_PER_BLOCK_KER2, TILE_SIZE * 2^LEVEL];
        kernel_2.GridSize = ker2Grid;

        tic;
        ker2_result = feval(kernel_2, im, ker2_result, im_height, im_width, TILE_SIZE * 2^LEVEL);
        ker2_result = gather(ker2_result);
        
        if n > n_max/10
            kernel_2_time = kernel_2_time+(toc-kernel_2_time)/kernel_2_iter;
            kernel_2_iter = kernel_2_iter+1;
        end
        
        LEVEL = LEVEL + 1;
        ker2Grid = ker2Grid / 2;

        ker2Outputs(:, :, i) = ker2_result;
        i = i + 1;
    end
    

    %% kernel four - final update

    kernel_4.ThreadBlockSize = [TILE_SIZE, TILE_SIZE];
    kernel_4.GridSize = [ceil(im_width/TILE_SIZE), ceil(im_height/TILE_SIZE)];

    tic;
    ker4_result = zeros(size(im));
    ker4_result = feval(kernel_4, ker2_result, ker4_result, im_height, im_width);
    ker4_result = gather(ker4_result);

    if n > n_max/10
        kernel_4_time = kernel_4_time+(toc-kernel_4_time)/kernel_4_iter;
        kernel_4_iter = kernel_4_iter+1;
    end
    
    if time_measure == false
        figure();
        imshow(ker4_result, []);
        title('kernel 4 result');
    end
end

if time_measure == true
    kernel_1_time
    kernel_2_time
    kernel_2_totaltime = kernel_2_time*kernel_2_iter/kernel_4_iter
    kernel_4_time 
end

function [resizedImg, levels] = preprocessImage(img)

    im_height = size(img, 1);
    im_width = size(img, 2);  

    pow = ceil(log2(max(im_height, im_width)));
    width = 2^pow;

    resizedImg = zeros(width);
    resizedImg(1:im_height, 1:im_width) = img;
    levels = 0;
end
