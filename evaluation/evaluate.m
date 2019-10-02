clear; clc;
addpath(genpath('util'));

works    = {'FC2N', 'FC2N+', 'FC2N++'};
datasets = {'Set5', 'Set14', 'B100', 'Urban100', 'Manga109'};
%datasets = {'Set5'};
apath    = '../works';
ext      = '*.png';
scales   = {2, 3, 4};

EvalTable = zeros(length(works), length(datasets), length(scales), 2);
for w = 1:length(works)
    work = works{w};
    fprintf('================ [ %s ] =================\n', work);
    
    for d = 1:length(datasets)
        dataset = datasets{d};
        fprintf('%s: \n', dataset);
    
        for s = 1:3
            scale = scales{s};
            fprintf('\tX%d: ', scale);
            hDir = fullfile('../datasets', dataset, ['image_SRF_' num2str(scale)], 'HR');
            sDir = fullfile(apath, work, dataset, ['X' num2str(scale)]);
            him_items = dir(fullfile(hDir, ext));
            sim_items = dir(fullfile(sDir, ext));
            num_image = length(him_items);
            assert(length(sim_items) == num_image, 'image number incompatible!');
        
            mean_psnr = 0.0;
            mean_ssim = 0.0;
            mean_ifc  = 0.0;
            for i = 1:num_image
                fprintf('%03d/%03d\n', i, num_image);
                him_name = him_items(i).name;
                sim_name = sim_items(i).name;
                imGT = imread(fullfile(hDir, him_name));
                imSR = imread(fullfile(sDir, sim_name));
            
                % calc evaluation metrics
                [~, cur_psnr, cur_ssim] = compute_diff(imGT, imSR, scale);
                mean_psnr = mean_psnr + cur_psnr;
                mean_ssim = mean_ssim + cur_ssim;
                fprintf('\b\b\b\b\b\b\b\b');
            end
            mean_psnr = mean_psnr / num_image;
            mean_ssim = mean_ssim / num_image;
            fprintf('%.2f/%.4f\n', mean_psnr, mean_ssim);
            EvalTable(w, d, s, 1) = mean_psnr;
            EvalTable(w, d, s, 2) = mean_ssim;
        end
    
    end
end

%save(fullfile('./psnr_ssim_ifc.mat'), 'EvalTable');

