close all;
clear all;

size = 256;

img = zeros(size, size, 3);
offset = 0;

for row = 1:size

    if offset == 0
        offset = 3;
        draw0 = 0;
        draw1 = 1;
    elseif offset == 3
        offset = 1;
        draw0 = 3;
        draw1 = 4;
    elseif offset == 1
        offset = -1;
        draw0 = 1;
        draw1 = 2;
    elseif offset == -1
        offset = 2;
        draw0 = 0;
        draw1 = 4;
    elseif offset == 2
        offset = 0;
        draw0 = 2;
        draw1 = 3;
    end
    for col = 1:5:size
       
        if col+draw0 <= size
            img(row,col+draw0,:) = 255;
        end
        if col+draw1 <= size
            img(row,col+draw1,:) = 255;
        end
       

   end

end

figure();
imshow(img);
finalMat = img;
imwrite(finalMat,['../test_images/schody', num2str(size),'.bmp'],'bmp');