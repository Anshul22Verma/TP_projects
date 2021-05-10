import argparse
import base64
import json
import os
import numpy as np
import os.path as osp
import cv2
import imgviz
import PIL.Image
from tqdm import tqdm

from labelme import utils


class FindJPreprocess:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.desired_size = (256, 256)
        self.raw_mask_dir = osp.join(osp.dirname(raw_dir), 'raw_mask')
        os.listdir(self.raw_mask_dir)
        self.input_dir = osp.join(osp.dirname(raw_dir), 'img')
        self.json_mask_dir = osp.join(osp.dirname(self.input_dir), 'json_mask')
        self.mask_dir = osp.join(osp.dirname(self.input_dir), 'mask')
        self.viz_dir = osp.join(osp.dirname(self.input_dir), 'viz_mask')

        # making the directories for masks if they do not exist
        if not osp.exists(self.viz_dir):
            os.mkdir(self.viz_dir)
        if not osp.exists(self.mask_dir):
            os.mkdir(self.mask_dir)

    def resize_images(self):
        for img in os.listdir(self.raw_dir):
            img_ = cv2.imread(osp.join(self.raw_dir, img))
            cv2.imwrite(osp.join(self.input_dir, img), cv2.resize(img_, self.desired_size))
        return None

    def create_masks(self):
        # I only have png-input images at the moment
        for img in tqdm(os.listdir(self.input_dir)):
            if img.replace('png', 'json') in os.listdir(self.json_mask_dir):
                self.reformat_images(img.replace('png', 'json'))
            elif img in os.listdir(self.raw_mask_dir):
                self.reformat_mask(img)
            else:
                self.save_empty_mask(img)
        return None

    def reformat_images(self, json_file):
        data = json.load(open(osp.join(self.json_mask_dir, json_file)))
        imageData = data.get("imageData")

        if not imageData:
            imagePath = os.path.join(self.input_dir, data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {"_background_": 0, "J": 255, "no_J": 0}#, "someone": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl_viz = imgviz.label2rgb(
            label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
        )
        # To save masks
        cv2.imwrite(osp.join(self.mask_dir, f"{json_file.replace('json', 'png')}"), lbl[..., np.newaxis])
        # for visualization
        PIL.Image.fromarray(lbl_viz).save(osp.join(self.viz_dir, f"label_viz_{json_file.replace('json', 'png')}"))
        return None

    def reformat_mask(self, img_file):
        mask = cv2.imread(osp.join(self.raw_mask_dir, img_file), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        cv2.imwrite(osp.join(self.mask_dir, img_file), cv2.resize(mask, self.desired_size))

    def save_empty_mask(self, img):
        img_ = cv2.imread(osp.join(self.input_dir, img), cv2.COLOR_BGR2RGB)
        mask_ = np.zeros(np.shape(img_))
        # saving mask as a gray image
        cv2.imwrite(osp.join(self.mask_dir, img), mask_[:, :, 0])
        # creating label visualization and saving it
        cv2.imwrite(osp.join(self.viz_dir, f"label_viz_{img}"), cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY))
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()

    input_dir = args.input_dir

    # initialize and create masks for where J exists
    preproc = FindJPreprocess(input_dir)
    # Create masks
    preproc.resize_images()
    preproc.create_masks()