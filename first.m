close all
clear all

cudaFilename = 'first_kernel.cu';
ptxFilename = 'first_kernel.ptx';
kernel = parallel.gpu.CUDAKernel( ptxFilename, cudaFilename );

im = imread('peppers.png');
im = rgb2gray(im);

im_height = size(im,1);
im_width = size(im,2);
block_width = 16;
block_height = 16;
kernel.ThreadBlockSize = [block_width, block_height];
kernel.GridSize = [ceil(im_height/block_height), ceil(im_width/block_width)];

mask = [-0.112737 0 -0.112737; -0.274526 0 0.274526; -0.112737 0 0.112737];
result = gpuArray(zeros(size(im)));
result = feval(kernel, im, mask, result, im_width, im_height);

figure;imshow(result,[])

test = conv2(im,mask);
figure;imshow(im,[]);
