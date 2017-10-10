function results = TCCF(seq)
cleanupObj = onCleanup(@cleanupFun);
rand('state', 0);
close all;

addpath('caffe/matlab/', 'util');
addpath('util');
data_path = [seq.path seq.name '/'];

%% init caffe 
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

%% VGG 16 layers model
feature_solver_def_file = 'model/feature_solver.prototxt';
model_file = 'model/VGG_ILSVRC_16_layers.caffemodel';
fsolver = caffe.Solver(feature_solver_def_file);
fsolver.net.copy_from(model_file);

%% prepare some parameters for location estimation
roi_size = 298;
num_z = 4;
im_name = sprintf ([data_path 'img/%0' num2str(num_z) 'd.jpg'], seq.startFrame);
im_sz = size(imread(im_name));
cell_size = 8;
compute_size = floor([roi_size,roi_size] / cell_size);%(width,height)

A = 0.011;%relaxed factor
lambda = 1e-4;%regularization
location_sigma_factor = 0.1;%spational bandwidth (proportioal to targetsize)
inter_factor = 0.00902; %to update the filter

location_layers = {'conv4_1','conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'};%layers to get features for location estimation
%% prepare some parameters for scale estiamtion
L = 33;
number_of_scale = 33;
scale_step = 1.02;
scale_sigma_factor = 1/4;
scale_model_max_area = 512;%the maximum size of scale examples
ss = ceil(number_of_scale/2) - (1:number_of_scale);
scale_factors = scale_step.^ss;
%scale filter cosin window
scale_window = single(hann(L));
scale_window = scale_window((L - number_of_scale)/2 + 1 : (L + number_of_scale)/2);

%define the resize dimension used for features extraction in the scale estimation
scale_model_factor = 1;
if prod(seq.init_rect(3:4)) > scale_model_max_area
    scale_model_factor = sqrt(scale_model_max_area/prod(seq.init_rect(3:4)));
end
scale_model_size = floor(seq.init_rect(4:-1:3) * scale_model_factor);
current_scale_factor = 1;
%find maximum and minimum scales
min_scale_factor = scale_step ^ ceil(log(max(5./ (2 * seq.init_rect(3:4)))) / log(scale_step));
max_scale_factor = scale_step ^ floor(log(min([im_sz(2),im_sz(1)] ./ seq.init_rect(3:4))) / log(scale_step));
%desired scale filter output (gaussian shaped),bandwidth proportional to number of scales
scale_sigma = 11.75 * scale_sigma_factor; 
scale_y = exp(-0.5 * (ss.^2) / scale_sigma^2);
scale_yf = single(fft(scale_y));


%% init
target_center = zeros(seq.endFrame - seq.startFrame + 1,2);
target_center(1,:) = [seq.init_rect(1) + floor(seq.init_rect(3)/2) - 1, seq.init_rect(2) + floor(seq.init_rect(4)/2) - 1];%(x,y)
position_box = zeros(seq.endFrame - seq.startFrame + 1, 4);
position_box(1,:) = seq.init_rect;%(x,y,w,h)

%%
tic;

for frame = seq.startFrame:seq.endFrame
    
    fprintf([seq.name ' %d' ' / ' '%d \n'],frame,seq.endFrame);
      
    % read images
    im_name = sprintf([data_path 'img/%0' num2str(num_z) 'd.jpg'], frame);
    
    if strcmpi('David',seq.name)
        im_name = sprintf([data_path 'img/%0' num2str(num_z) 'd.jpg'], frame+299);%for David
    end
    
    %im_name = sprintf([data_path 'img/%0' num2str(num_z) 'd.jpg'], 1);
    im = double(imread(im_name));
    if size(im,3)~=3
        im(:,:,2) = im(:,:,1);
        im(:,:,3) = im(:,:,1);
    end
    
    if frame > seq.startFrame 
        %% location estimation
        location_window_size = get_search_window_size(position_box(frame - 1,3:4), im_sz); %to extract roi for location estimation.(width,height),
 
        % extract roi
        location_roi = get_location_roi(im, target_center(frame -1,:), location_window_size);
        location_features = get_location_features(location_roi,roi_size, fsolver,location_layers, compute_size);
        for i = 1 : length(location_layers)
            location_zf = fft2(location_features{i});%a new testing data, we first transform it to Fourier domain, Eq.(4)
            location_response{i} = real(fftshift(ifft2(sum(location_filter_num{i} .* location_zf, 3) ./ (location_filter_den{i} + lambda))));%Eq.(6)
            location_response{i} = imresize(location_response{i},location_window_size);
            max_response(i) =  max(location_response{i}(:));
            [col, row] = find(location_response{i} == max_response(i),1);
            expert{i}.col = col;
            expert{i}.row = row;
        end 
        col = 0;
        row = 0;
        for i = 1 : length(location_layers)
            col = col + location_w(i) * expert{i}.col;
            row = row + location_w(i) * expert{i}.row;
        end
        col = round(col);
        row = round(row);

        delta_x = col -floor(location_window_size(1)/2);
        delta_y = row -floor(location_window_size(2)/2);

        target_center(frame,:) = target_center(frame - 1,:) + [delta_x - 1, delta_y - 1];
        target_center(frame,1) = max(0,target_center(frame,1));
        target_center(frame,1) = min(size(im,2),target_center(frame,1));
        target_center(frame,2) = max(0,target_center(frame,2));
        target_center(frame,2) = min(size(im,1),target_center(frame,2));
        position_box(frame,:) = [target_center(frame,:) - position_box(frame - 1,3:4)/2 + 1,position_box(frame-1,3:4)];
        
        % update location filter weight
        for i = 1 : length(location_layers)           
            loss(6, i) = max_response(i) - location_response{i}(col,row);%'6' means we use 6 frames to update weights
        end
        loss_mean = mean(loss(1:5, :));
        loss_standard = std(loss(1:5, :));
        loss_mean(loss_mean < 0.0001) = 0;
        loss_standard(loss_standard < 0.0001) = 0;
        
        current_diff = loss_mean - loss(6,:);
        alpha = 0.97 * exp(-10 * abs(current_diff) ./ (loss_standard + eps));
        %truncation
        alpha(alpha > 0.9) = 0.97;
        alpha(alpha < 0.12) = 0.119;
        loss_average = sum(location_w .* loss(6,:));
        R = R .* (alpha) + (1 - alpha) .* (loss_average - loss(6,:));
        %update loss history
        loss_index = mod(frame - 1, 5) + 1;
        loss(loss_index,:) = loss(6,:);
        
        c = find_nh_scale(R, A);
        location_w = nnhedge_weights(R, c, A);
        location_w = location_w / sum(location_w);   
        
        %% scale estimation
        scale_x = get_scale_sample(im, target_center(frame,2:-1:1), seq.init_rect(4:-1:3), current_scale_factor * scale_factors, scale_window, scale_model_size);
        scale_xf = fft(scale_x,[],2);
        scale_response = real(ifft(sum(scale_filter_num .* scale_xf,1) ./(scale_filter_den + lambda)));
        recovered_scale = find(scale_response == max(scale_response(:)),1);
        current_scale_factor = current_scale_factor * scale_factors(recovered_scale);
        if current_scale_factor < min_scale_factor
            current_scale_factor = min_scale_factor;
        elseif current_scale_factor > max_scale_factor
            current_scale_factor = max_scale_factor;
        end
        position_box(frame,3:4) = seq.init_rect(3:4) * current_scale_factor;
        position_box(frame,:) = [target_center(frame,:) - position_box(frame,3:4)/2 + 1,position_box(frame,3:4)];

        
    end 
    
    %scale estimation 
    scale_x = get_scale_sample(im, target_center(frame,2:-1:1), seq.init_rect(4:-1:3), current_scale_factor * scale_factors, scale_window, scale_model_size);
    scale_xf = fft(scale_x,[],2);
    new_scale_filter_num = bsxfun(@times,scale_yf,conj(scale_xf));
    new_scale_filter_den = sum(scale_xf .* conj(scale_xf),1);
    
    %desired location filter output (gaussian shaped0, bandwidth proportional to target size
    location_window_size = get_search_window_size(position_box(frame,3:4), im_sz); %(width,height)
    output_sigma = sqrt(prod(position_box(frame,3:4))) * location_sigma_factor / cell_size;
    location_yf = fft2(gaussian_shaped_labels(output_sigma, compute_size));
   
    location_roi = get_location_roi(im, target_center(frame,:), location_window_size);
    location_features = get_location_features(location_roi,roi_size,fsolver,location_layers, compute_size);
   
    for i = 1:length(location_layers)
        location_xf{i} = fft2(location_features{i});
        new_location_filter_num{i} = bsxfun(@times, location_yf,conj(location_xf{i}));
        new_location_filter_den{i} = sum(location_xf{i} .* conj(location_xf{i}), 3);
    end
  
    if frame == seq.startFrame %first frame, train with a single image
        %scale filter
        scale_filter_num = new_scale_filter_num;
        scale_filter_den = new_scale_filter_den;
        
        % location filter
        for i = 1:length(location_layers)
            location_filter_num{i} = new_location_filter_num{i};
            location_filter_den{i} = new_location_filter_den{i};
        end

        location_w = [1, 0.2, 0.2, 0.02, 0.03, 0.01];%initial weigth of different layers
        location_w = location_w / sum(location_w);
        R(1:length(location_layers)) = 0;
        loss = zeros(6,length(location_layers));
    else
        % update scale filter
        scale_filter_num = (1 - inter_factor) * scale_filter_num + inter_factor * new_scale_filter_num;
        scale_filter_den = (1 - inter_factor) * scale_filter_den + inter_factor * new_scale_filter_den;
        % update location filter
        for i = 1:length(location_layers)
            location_filter_num{i} = (1 - inter_factor) * location_filter_num{i} + inter_factor * new_location_filter_num{i};
            location_filter_den{i} = (1 - inter_factor) * location_filter_den{i} + inter_factor * new_location_filter_den{i};
        end          
    end
    
    if seq.visualization ==1
    % Draw resutls
        if frame == seq.startFrame,  %first frame, create GUI
            figure('Name','Tracking Results');
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position', position_box(frame,:), 'EdgeColor','r', 'linewidth', 2);
            text_handle = text(10, 10, sprintf('#%d / %d',frame, seq.endFrame));
            set(text_handle, 'color', [1 1 0], 'fontsize', 16, 'fontweight', 'bold');
        
        else
            set(im_handle, 'CData', uint8(im));
            set(rect_handle, 'Position', position_box(frame,:));
            set(text_handle, 'string', sprintf('#%d / %d',frame, seq.endFrame));
        end 
            drawnow
    end

end

t_total = toc;

results.type = 'rect';
results.res = position_box; 
results.annoBegin = 1;
results.startFrame = seq.startFrame;
if strcmpi('David',seq.name)
    results.startFrame = seq.startFrame+299;%for David
end
results.len = seq.endFrame - seq.startFrame + 1;
results.fps = (seq.endFrame - seq.startFrame + 1) / t_total;
end  
    
    
