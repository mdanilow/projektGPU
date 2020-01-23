close all;
clear all;

size = 69;

img = zeros(size, size, 3);

%% slimak
leng = size;
% % % % % % dir:
% % % % % % right = 1;
% % % % % % up = 2;
% % % % % % left = 3;
% % % % % % down = 4;
img(:,1,:) = 255;
start = [size, 1];
while(leng > 2)
    k = 0;
    for dir = 1:4

        switch dir
            
            case 1
                for i = start(2):leng + start(2)-1
                    img(start(1),i,:) = 255;
                end
                k = k + 1;
                if k == 2
                   leng = leng - 2;
                   k = 0;
                end
                start(2) = leng + start(2)-1;

            case 2
                for i = start(1):-1:(start(1)-leng+2)
                    
                    display(i);
                    img(i, start(2),:) = 255; 
                end
                k = k + 1;
                if k == 2
                   leng = leng - 2;
                   k = 0;
                end
                start(1) = start(1)-leng;
                
            case 3
                for i = start(2):-1:(start(2)-leng+1)
                     img(start(1),i,:) = 255; 
                end
                k = k + 1;
                if k == 2
                   leng = leng - 2;
                   k = 0;
                end
                start(2) = start(2)-leng+1;
                
            case 4
                for i = start(1):start(1)+leng-2
                     img(i, start(2),:) = 255;
                end
                k = k + 1;
                if k == 2
                   leng = leng - 2;
                   k = 0;
                end
                start(1) = start(1)+leng;
            otherwise
                disp('error');
                
        end
    
    end
    

end




figure();
imshow(img);
finalMat = img;
imwrite(finalMat,['../test_images/slimak', num2str(size),'.bmp'],'bmp');