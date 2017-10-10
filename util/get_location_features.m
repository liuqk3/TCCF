function features = get_location_features(roi,roi_size, feature_solver, layers, compute_size)

roi = imresize(roi,[roi_size,roi_size]);
roi = impreprocess(roi);
%feature_solver.net.set_net_phase('test');
feature_input = feature_solver.net.blobs('data');
%feature_solver.net.set_input_dim([0, 1, 3, size(roi,1),size(roi,2)]);
feature_input.set_data(single(roi));
feature_solver.net.forward_prefilled();

features = cell(1,length(layers));

cos_window = single(hann(compute_size(1)) * hann(compute_size(2))');
for i =1:length(layers)
    feature_blob = feature_solver.net.blobs(layers{i});
    one_features = feature_blob.get_data();
    one_features = imresize(one_features, [compute_size(1),compute_size(2)]);
    features{i} = bsxfun(@times, one_features,cos_window);
end

end