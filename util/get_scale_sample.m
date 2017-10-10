function out = get_scale_sample(im, pos, base_target_sz, scale_factors, scale_window, scale_model_sz)

% out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)
% 
% Extracts a sample for the scale filter at the current
% location and scale.
% im : the image
% pos : target center cordinate,(y,x)
% base_target_sz : the initial target size
% scale_factors : the factors to get scale samples
% scale_window : hanning widow to constraint scale samples
% scale_model_sz : equal to 

nScales = length(scale_factors);

for s = 1:nScales
    patch_sz = floor(base_target_sz * scale_factors(s));
    
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
    
    % check for out-of-bounds coordinates, and set them to the values at
    % the borders
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    % extract image
    im_patch = im(ys, xs, :);
    
    % resize image to model size
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
    % extract scale features
    temp_hog = fhog(single(im_patch_resized), 4);
    temp = temp_hog(:,:,1:31);
     
    
    if s == 1
        out = zeros(numel(temp), nScales, 'single');
    end
    
    % window
    out(:,s) = temp(:) * scale_window(s);
    %size_out = size(out(:,s))
end