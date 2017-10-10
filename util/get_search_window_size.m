function window_sz = get_search_window_size( target_sz, im_sz)
 
    %window_size ,(width, height)
    %im_sz, (height,width,channel)
    
    % extra area surrounding the target
    padding = struct('generic', 1.4, 'large', 1, 'height', 0.4);
    % For objects with large height, we restrict the search window with padding.height
    if target_sz(2)/ target_sz(1)>2   
        window_sz = floor(target_sz.*[1+padding.generic, 1+padding.height]);
        %state = 'height'
    % For objects with large height and width and accounting for at least 10 percent of the whole image,
    % we only search 2x height and width
    elseif prod(target_sz)/prod(im_sz(1:2))>0.05 %if the area of target ocupies 5% of the image
        window_sz=floor(target_sz*(1+padding.large));  
        %state = 'large'
        
    %otherwise, we use the padding configuration    
    else        
        window_sz = target_sz * (1 + padding.generic);
        %state = 'generic'
    end
end