clear all;
close all;

img = imread('coins.png');
img = uint8(imbinarize(img) .* 255);
img = imfill(img);
imshow(img);

width = size(img, 2);
height = size(img, 1);
labels = zeros(size(img));

for row = 1:height
    for col = 1:width

        labels(row, col) = to1DAddress(row, col, height);
        [nhood, quantity] = getNeighbours(row, col, img);

        for n = 1:quantity
            nRow = nhood(n, 1);
            nCol = nhood(n, 2);
            if(img(row, col) == img(nRow, nCol))
                
            end
        end
        
    end
end


function addr = to1DAddress(row, col, height)

    addr = (row-1) * height + col;
end


function [neighbours, quantity] = getNeighbours(row, col, img)

    width = size(img, 2);
    height = size(img, 1);
    neighbours = zeros(8, 2);

    if(row == 1)

        if(col == 1)
            quantity = 0;
        else
            quantity = 1;
            neighbours(1, :) = [row, col-1];
        end

    elseif(row == height)

        if(col == 1)
            quantity = 2;
            neighbours(1, :) = [row-1, col];
            neighbours(2, :) = [row-1, col+1];
        elseif(col == width)
            quantity = 3;
            neighbours(1, :) = [row, col-1];
            neighbours(2, :) = [row-1, col-1];
            neighbours(3, :) = [row-1, col];
        else
            quantity = 4;
            neighbours(1, :) = [row, col-1];
            neighbours(2, :) = [row-1, col-1];
            neighbours(3, :) = [row-1, col];
            neighbours(4, :) = [row-1, col+1];
        end

    elseif(col == 1)

        quantity = 2;
        neighbours(1, :) = [row-1, col];
        neighbours(2, :) = [row-1, col+1];

    elseif(col == width)

        quantity = 3;
        neighbours(1, :) = [row, col-1];
        neighbours(2, :) = [row-1, col-1];
        neighbours(3, :) = [row-1, col];
            
    else

        quantity = 4;
        neighbours(1, :) = [row, col-1];
        neighbours(2, :) = [row-1, col-1];
        neighbours(3, :) = [row-1, col];
        neighbours(4, :) = [row-1, col+1];
    end

end