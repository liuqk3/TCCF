%% demo_TCCF 
clear all;
close all;

show_seqs ={'DragonBaby'};

for i = 1:length(show_seqs)

    seq_name = show_seqs{i};
    data_path = './Dataset/';
    GT = load([data_path seq_name '/groundtruth_rect.txt']);
    startFrame = 1;
    endFrame = length(GT);

    seq.name = seq_name;
    seq.path = data_path;
    seq.startFrame = startFrame;
    seq.endFrame = endFrame;
    seq.init_rect = GT(1,:);
    seq.visualization = 1;
    results = TCCF(seq);
    results = {results};
    save(['results/' [lower(seq.name) '_TCCF.mat']],'results');
    
end



