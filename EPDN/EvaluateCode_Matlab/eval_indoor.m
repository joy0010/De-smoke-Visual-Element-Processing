clc
clear   

gt_PATH = 'indoor/gt/';        % dir storing ground truth images
pred_PATH = 'indoor_result/images/';          % dir storing generated images

gt_Dir = dir([gt_PATH '*.png']); 
pred_Dir = dir([pred_PATH '*.png']);     

length(gt_Dir)  

len_pred_Dir = length(pred_Dir)

PSNRs = zeros([1,len_pred_Dir]);  % allocate memory for psnr data
SSIMs = zeros([1,len_pred_Dir]);  % allocate memory for ssim data


for i = 1:length(gt_Dir)  

    gt = imread([gt_PATH gt_Dir(i).name]);  % read gt image
    gt = imcrop(gt,[17,17,607,447]);        % crop image to the same size
    % gt = imcrop(gt,[11,11,619,459]);        % crop image to the same size
%     gt = imcrop(gt,[9,9,623,463]);         % single scale
    % size(gt)
    % imshow(gt)
    
    for j = 1:10
        %sprintf('gt file: %s\n', gt_Dir(i).name)
        
        index = 10*(i-1)+j;
        pred_file = pred_Dir(index).name;
        pred = imread([pred_PATH pred_file]);  % read generated image
        % size(pred)

        % calculate
        [ssimval,~] = ssim(gt,pred);
        
        PSNRs(index) = psnr(gt, pred); 
        SSIMs(index) = ssimval;
        sprintf('%d processing: %s\n psnr: %.4f', index, pred_file, PSNRs(index))
        

    end
end

sprintf('%d processing: %s\n psnr: %.4f', i, pred_Dir(i).name, PSNRs(i))
PSNR = sum(PSNRs)/len_pred_Dir
SSIM = sum(SSIMs)/len_pred_Dir


