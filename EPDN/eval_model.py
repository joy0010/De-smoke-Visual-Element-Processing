from util import images as util
import os
import cv2

# HR_path = 'dataset/benchmark/Set5/HR'
# SR_path = 'experiments/results/Set5/x4'
# HR_path = 'dataset/benchmark/Urban100/HR'

# SR_path = 'experiments/results/Urban100/x4'
# SR_path = 'experiments/results/csnla_Urban100'

# HR_path = 'datasets/mytest_A/gt'
# SR_path = 'Pix2PixHD_modify/results/p2p/mytest_16/images'

HR_path = 'datasets/testdata/indoor/gt'
SR_path = 'Pix2PixHD_modify/results/official_pretrained/testdata0/indoor/hazy_16/images'

n_channels = 3

def evulate():
    print("Start...")
    print(HR_path)
    print(SR_path)

    hr_paths = util.get_image_paths(HR_path)
    numbers = len(hr_paths)
    sum_psnr = 0
    max_psnr = 0
    min_psnr = 100
    sum_ssim = 0
    max_ssim = 0
    min_ssim = 1
    for hr_path in hr_paths:
        # img_name, ext = os.path.splitext(os.path.basename(img_path))
        img_name = os.path.basename(hr_path)
        sr_path = os.path.join(SR_path,img_name)
        print(img_name)

        img_Hr = util.imread_uint(hr_path, n_channels=n_channels)  # HR image, int8
        img_Sr = util.imread_uint(sr_path, n_channels=n_channels)  # HR image, int8
        psnr = util.calculate_psnr(img_Sr, img_Hr,)
        print(psnr)
        sum_psnr += psnr
        max_psnr = max(max_psnr,psnr)
        min_psnr = min(min_psnr, psnr)
        ssim = util.calculate_ssim(img_Sr, img_Hr,)
        # print(ssim)
        sum_ssim += ssim
        max_ssim = max(max_ssim,ssim)
        min_ssim = min(min_ssim, ssim)

    print('Average psnr = ', sum_psnr / numbers)
    print('min_psnr = ', min_psnr)
    print('Max_psnr = ', max_psnr)
    print('Average ssim = ', sum_ssim / numbers)
    print('min_ssim = ', min_ssim)
    print('Max_ssim = ', max_ssim)


def evulate_diff_name():
    print("Start...")
    print(HR_path)
    print(SR_path)

    hr_paths = util.get_image_paths(HR_path)
    numbers = len(hr_paths) * 10
    sum_psnr = 0
    max_psnr = 0
    min_psnr = 100
    sum_ssim = 0
    max_ssim = 0
    min_ssim = 1
    for hr_path in hr_paths:
        name, ext = os.path.splitext(os.path.basename(hr_path))
        # img_name = os.path.basename(hr_path)
        # print(img_name)
        print(hr_path)
        img_Hr_0 = util.imread_uint(hr_path, n_channels=n_channels)  # HR image, int8
        img_Hr = img_Hr_0[16:464, 16:624]
        # print(img_Hr.shape)
        for i in range(1,11):
            temp = str(name) + '_' + str(i) + '_final.png'
            # print(temp)
            sr_path = os.path.join(SR_path, temp)
            print(sr_path)

            img_Sr = util.imread_uint(sr_path, n_channels=n_channels)  # HR image, int8
            psnr = util.calculate_psnr(img_Sr, img_Hr,)
            # print(psnr)
            sum_psnr += psnr
            max_psnr = max(max_psnr,psnr)
            min_psnr = min(min_psnr, psnr)
            ssim = util.calculate_ssim(img_Sr, img_Hr,)
            # print(ssim)
            sum_ssim += ssim
            max_ssim = max(max_ssim,ssim)
            min_ssim = min(min_ssim, ssim)

    print('Average psnr = ', sum_psnr / numbers)
    print('min_psnr = ', min_psnr)
    print('Max_psnr = ', max_psnr)
    print('Average ssim = ', sum_ssim / numbers)
    print('min_ssim = ', min_ssim)
    print('Max_ssim = ', max_ssim)


if __name__ == '__main__':
    print('-------------------------compute psnr and ssim for evaluate model---------------------------------')
    # evulate()
    evulate_diff_name()



