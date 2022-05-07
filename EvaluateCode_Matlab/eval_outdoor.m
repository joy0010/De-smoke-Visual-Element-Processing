clc
clear       

gt_PATH = 'outdoor/gt/';        % dir storing ground truth images
pred_PATH = 'outdoor_result/images/';          % dir storing generated images

gt_Dir = dir([gt_PATH '*.png']);      % iter all .png files
pred_Dir = dir([pred_PATH '*.png']);  % iter all .png files

PSNRs = zeros([1,length(pred_Dir)]);  % allocate memory for psnr data
SSIMs = zeros([1,length(pred_Dir)]);  % allocate memory for ssim data

for i = 1:length(gt_Dir)
    
    gt = imread([gt_PATH gt_Dir(i).name]);        % read gt image
    pred = imread([pred_PATH pred_Dir(i).name]);  % read generated image
    
    % resize gt and pred to the same size
    [ny1, nx1, np1] = size(gt); 
    [ny2, nx2, np2] = size(pred); 
    nx = max(nx1,nx2); 
    ny = max(ny1,ny2); 
    gt = imresize(gt, [ny nx]);
    pred = imresize(pred, [ny nx]);
    
    % calculate
    [ssimval,~] = ssim(gt,pred);

    PSNRs(i) = psnr(gt, pred); 
    SSIMs(i) = ssimval;
    sprintf('%d processing: %s\n psnr: %.4f', i, pred_Dir(i).name, PSNRs(i))

end

sprintf('%d processing: %s\n psnr: %.4f', i, pred_Dir(i).name, PSNRs(i))
PSNR = sum(PSNRs)/length(pred_Dir)
SSIM = sum(SSIMs)/length(pred_Dir)


