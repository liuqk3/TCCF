function I = impreprocess(im)
% im's dimension is in height, width, channel order.
mean_pix = [103.939, 116.779, 123.68]; % BGR
im = permute(im, [2,1,3]);%change im's dimension to width, height, channel order.
im = im(:,:,3:-1:1);%change RGB to BGR
I(:,:,1) = im(:,:,1)-mean_pix(1); % substract mean
I(:,:,2) = im(:,:,2)-mean_pix(2);
I(:,:,3) = im(:,:,3)-mean_pix(3);
end
