function out = get_location_roi(im, pos, sz)
%GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   Returns sub-window of image IM centered at POS ([y, x] coordinates),
%   with size SZ ([height, width]). If any pixels are outside of the image,
%   they will replicate the values at the borders.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	if isscalar(sz)  %square sub-window
		sz = [sz, sz];
    end

    xs = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
	ys = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);

	
	%check for out-of-bounds coordinates, and set them to the values at
	%the borders
	xs(xs < 1) = 1;
	ys(ys < 1) = 1;
	xs(xs > size(im,2)) = size(im,2);%size(im,2) is width
	ys(ys > size(im,1)) = size(im,1);%size(im,1) is height
    xs = round(xs);
    ys = round(ys);
	
	%extract image
	out = im(ys, xs, :);
end

