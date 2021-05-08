'''
Lets implement CutAndPaste augmentation

This augmentations can be added as an augmentation in the DataGenerators, but for the sake of keeping this project
simple I am doing this separately and then performing other-augmentations.

This can be considered as the first augmentation of Albumentation augmentations.
ref : https://arxiv.org/pdf/2012.07177.pdf
LETSS DO IT !! CutAndPaste is no longer plagiarizing
'''

import cv2
import argparse
import base64
import json
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import os
import os.path as osp

from labelme import utils


class CopyAndPaste:
    def __init__(self, input_dir, background_dir):
        self.input_dir = input_dir
        self.json_mask_dir = osp.join(osp.dirname(self.input_dir), 'json_mask')
        self.mask_dir = osp.join(osp.dirname(self.input_dir), 'mask')
        self.background_dir = background_dir

        # default can be changed anytime
        self.augmentation_copies = 10
        self.w_J_test_size = 3
        self.wo_J_test_size = 2
        self.img_sz = 256

    def augment_images(self):
        # creating a random-test set for no leakage from training
        test_samples = []
        w_Js = [w_J for w_J in os.listdir(self.json_mask_dir) if 'no' not in w_J]
        wo_Js = [wo_J for wo_J in os.listdir(self.json_mask_dir) if 'no' in wo_J]
        test_samples += list(np.random.choice(w_Js, size=self.w_J_test_size, replace=False))
        test_samples += list(np.random.choice(wo_Js, size=self.wo_J_test_size, replace=False))

        imgs = []
        grps = []
        for img_f in tqdm(os.listdir(self.input_dir)):
            if 'CAP' not in img_f:
                if img_f in os.listdir(self.mask_dir):
                    imgs.append(img_f.replace('.json', ''))
                    if img_f not in test_samples:
                        grps.append('train')
                        img, mask = self.get_img_n_mask(img_f)
                        imgs, grps = self.create_augmentations(img, mask, imgs, grps,
                                                               img_name=img_f.replace('.png', ''))
                    else:
                        grps.append('test')
        df = pd.DataFrame()
        df['images'] = imgs
        df['group'] = grps
        df.to_csv(osp.join(osp.dirname(self.input_dir), 'log_meta.csv'), index=False)

    def get_img_n_mask(self, img_f):
        img_ = cv2.imread(osp.join(self.input_dir, img_f), cv2.COLOR_BGR2RGB)
        mask_ = cv2.imread(osp.join(self.mask_dir, img_f), cv2.IMREAD_GRAYSCALE)
        return img_, mask_

    def create_augmentations(self, img, mask, img_list, group_list, img_name):
        # first lets select 10-images at random from the background
        background_imgs = list(np.random.choice(os.listdir(self.background_dir),
                                                size=self.augmentation_copies,
                                                replace=False))
        for idx, background_img in enumerate(background_imgs):
            '''
                There are two ways of doing we can add J in the same location as it is in the original image
                but that noob-level lets resize the images before pasting them on top of background and then 
                un-masking the pizels which are not labeled as J.
            '''
            bg_img = cv2.imread(osp.join(self.background_dir, background_img), cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (self.img_sz, self.img_sz))
            if len(bg_img.shape) < 3:
                bg_img = np.repeat(bg_img[..., np.newaxis], 3, axis=2)

            # lets resize the og-image anywhere in between 180-256 (256 final desired size)
            random_sz = np.random.randint(180, 256)
            re_img = cv2.resize(img, (random_sz, random_sz))
            re_mask = cv2.resize(mask.astype('uint8'), (random_sz, random_sz))[..., np.newaxis]

            # now lets find a patch in the background-image
            x_init = np.random.randint(0, self.img_sz - random_sz)
            y_init = np.random.randint(0, self.img_sz - random_sz)
            bg_mask_img = np.zeros((self.img_sz, self.img_sz, 1))
            ix_, iy_, _ = np.where(re_mask != 0)
            bg_patch = bg_img[x_init:(x_init+random_sz), y_init:(y_init+random_sz), :]
            bg_patch[ix_, iy_, :] = re_img[ix_, iy_, :]

            bg_img[x_init:(x_init + random_sz), y_init:(y_init + random_sz), :] = bg_patch
            if 'no' not in img_name:
                bg_mask_img[x_init:(x_init + random_sz), y_init:(y_init + random_sz), :] = re_mask

            # saving the mask
            cv2.imwrite(osp.join(self.mask_dir, f'CAP_{img_name}_{idx}.png'), bg_mask_img)
            # saving the image
            cv2.imwrite(osp.join(self.input_dir, f'CAP_{img_name}_{idx}.png'), bg_img)
            img_list.append(f'CAP_{img_name}_{idx}')
            group_list.append('train')
        return img_list, group_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("background_dir")
    args = parser.parse_args()

    input_dir = args.input_dir
    background_dir = args.background_dir

    # initialize and create masks for where J exists
    augment = CopyAndPaste(input_dir, background_dir)
    # Create masks
    augment.augment_images()