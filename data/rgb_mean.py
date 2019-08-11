import os
import numpy as np
import cv2
import ipdb
def cal_mean():

    ims_path = '/home/liufc/JPEGImages/'  # 图像数据集的路径
    ims_list = os.listdir(ims_path)
    B_means = []
    G_means = []
    R_means = []
    for im_list in ims_list:
        im = cv2.imread(ims_path + im_list)
    
        # extrect value of diffient channel
        im_B = im[:, :, 0]
        im_G = im[:, :, 1]
        im_R = im[:, :, 2]
        # count mean for every channel
        im_B_mean = np.mean(im_B)
        im_G_mean = np.mean(im_G)
        im_R_mean = np.mean(im_R)
        # save single mean value to a set of means
        B_means.append(im_B_mean)
        G_means.append(im_G_mean)
        R_means.append(im_R_mean)
        print('图片：{} 的 BGR平均值为 \n[{}，{}，{}]'.format(im_list, im_B_mean, im_G_mean, im_R_mean))
    # three sets  into a large set
    a = [B_means, G_means, R_means]
    mean = [0, 0, 0]
    # count the sum of different channel means
    mean[0] = np.mean(a[0])
    mean[1] = np.mean(a[1])
    mean[2] = np.mean(a[2])
    print('数据集的BGR平均值为\n[{}，{}，{}]'.format(mean[0], mean[1], mean[2]))
  
    return mean #返回list
