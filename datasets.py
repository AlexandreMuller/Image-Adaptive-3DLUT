import random
import os
from pathlib import Path
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision_x_functional as TF_x


def _read_file_list(path):
    with open(path, 'r') as f:
        return sorted([line.strip() for line in f if line.strip()])


class ImageDataset_sRGB(Dataset):
    def __init__(self, root, mode="train", combined=True):
        self.mode = mode
        root = Path(root)

        def build_pairs(names):
            inputs = [str(root / "input" / "JPG/480p" / (n + ".jpg")) for n in names]
            experts = [str(root / "expertC" / "JPG/480p" / (n + ".jpg")) for n in names]
            return inputs, experts

        set1_inputs, set1_experts = build_pairs(_read_file_list(root / 'train_input.txt'))
        set2_inputs, set2_experts = build_pairs(_read_file_list(root / 'train_label.txt'))
        self.test_inputs, self.test_experts = build_pairs(_read_file_list(root / 'test.txt'))

        if combined:
            self.train_inputs = set1_inputs + set2_inputs
            self.train_experts = set1_experts + set2_experts
        else:
            self.train_inputs = set1_inputs
            self.train_experts = set1_experts

    def __getitem__(self, index):
        if self.mode == "train":
            inp_path = self.train_inputs[index % len(self.train_inputs)]
            exp_path = self.train_experts[index % len(self.train_experts)]
        else:
            inp_path = self.test_inputs[index % len(self.test_inputs)]
            exp_path = self.test_experts[index % len(self.test_experts)]

        img_name = os.path.basename(inp_path)
        img_input = Image.open(inp_path).convert('RGB')
        img_exptC = Image.open(exp_path).convert('RGB')

        if self.mode == "train":
            W, H = img_input.size
            crop_h = round(H * np.random.uniform(0.6, 1.0))
            crop_w = round(W * np.random.uniform(0.6, 1.0))
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            img_input = TF.adjust_brightness(img_input, np.random.uniform(0.8, 1.2))
            img_input = TF.adjust_saturation(img_input, np.random.uniform(0.8, 1.2))

        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.train_inputs)
        return len(self.test_inputs)


class ImageDataset_XYZ(Dataset):
    def __init__(self, root, mode="train", combined=True):
        self.mode = mode
        root = Path(root)

        def build_pairs(names):
            inputs = [str(root / "input" / "PNG/480p_16bits_XYZ_WB" / (n + ".png")) for n in names]
            experts = [str(root / "expertC" / "JPG/480p" / (n + ".jpg")) for n in names]
            return inputs, experts

        set1_inputs, set1_experts = build_pairs(_read_file_list(root / 'train_input.txt'))
        set2_inputs, set2_experts = build_pairs(_read_file_list(root / 'train_label.txt'))
        self.test_inputs, self.test_experts = build_pairs(_read_file_list(root / 'test.txt'))

        if combined:
            self.train_inputs = set1_inputs + set2_inputs
            self.train_experts = set1_experts + set2_experts
        else:
            self.train_inputs = set1_inputs
            self.train_experts = set1_experts

    def __getitem__(self, index):
        if self.mode == "train":
            inp_path = self.train_inputs[index % len(self.train_inputs)]
            exp_path = self.train_experts[index % len(self.train_experts)]
        else:
            inp_path = self.test_inputs[index % len(self.test_inputs)]
            exp_path = self.test_experts[index % len(self.test_experts)]

        img_name = os.path.basename(inp_path)
        img_input = np.array(cv2.imread(inp_path, -1))
        img_exptC = Image.open(exp_path).convert('RGB')

        if self.mode == "train":
            W, H = img_exptC.size
            crop_h = round(H * np.random.uniform(0.6, 1.0))
            crop_w = round(W * np.random.uniform(0.6, 1.0))
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            img_input = TF_x.adjust_brightness(img_input, np.random.uniform(0.6, 1.4))

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.train_inputs)
        return len(self.test_inputs)


class ImageDataset_sRGB_unpaired(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode
        root = Path(root)

        def build_pairs(names):
            inputs = [str(root / "input" / "JPG/480p" / (n + ".jpg")) for n in names]
            experts = [str(root / "expertC" / "JPG/480p" / (n + ".jpg")) for n in names]
            return inputs, experts

        self.set1_inputs, self.set1_experts = build_pairs(_read_file_list(root / 'train_input.txt'))
        self.set2_inputs, self.set2_experts = build_pairs(_read_file_list(root / 'train_label.txt'))
        self.test_inputs, self.test_experts = build_pairs(_read_file_list(root / 'test.txt'))

    def __getitem__(self, index):
        if self.mode == "train":
            inp_path = self.set1_inputs[index % len(self.set1_inputs)]
            exp_path = self.set1_experts[index % len(self.set1_experts)]
            seed = random.randint(1, len(self.set2_experts))
            img2 = Image.open(self.set2_experts[(index + seed) % len(self.set2_experts)]).convert('RGB')
        else:
            inp_path = self.test_inputs[index % len(self.test_inputs)]
            exp_path = self.test_experts[index % len(self.test_experts)]

        img_name = os.path.basename(inp_path)
        img_input = Image.open(inp_path).convert('RGB')
        img_exptC = Image.open(exp_path).convert('RGB')

        if self.mode == "test":
            img2 = img_exptC

        if self.mode == "train":
            W, H = img_input.size
            W2, H2 = img2.size
            crop_h = min(round(H * np.random.uniform(0.6, 1.0)), H2)
            crop_w = min(round(W * np.random.uniform(0.6, 1.0)), W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF.hflip(img2)

            img_input = TF.adjust_brightness(img_input, np.random.uniform(0.6, 1.4))
            img_input = TF.adjust_saturation(img_input, np.random.uniform(0.8, 1.2))

        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_inputs)
        return len(self.test_inputs)


class ImageDataset_XYZ_unpaired(Dataset):
    def __init__(self, root, mode="train", unpaird_data="fiveK"):
        self.mode = mode
        self.unpaird_data = unpaird_data

        file = open(os.path.join(root,'train_input.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train_label.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",set2_input_files[i][:-1] + ".png"))
            self.set2_expert_files.append(os.path.join(root,"expertC","JPG/480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"expertC","JPG/480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            seed = random.randint(1,len(self.set2_expert_files))
            img2 = Image.open(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img2 = img_exptC

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":
            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            W2,H2 = img2._size
            crop_h = min(crop_h,H2)
            crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF.hflip(img2)

            a = np.random.uniform(0.6,1.4)
            img_input = TF_x.adjust_brightness(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_HDRplus(Dataset):
    def __init__(self, root, mode="train", combined=True):
        self.mode = mode

        file = open(os.path.join(root,'train.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"middle_480p",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"output_480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"middle_480p",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"output_480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":

            ratio = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio)
            crop_w = round(W*ratio)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            try:
                img_input = TF_x.crop(img_input, i, j, h, w)
            except:
                print(crop_h,crop_w,img_input.shape())
            img_exptC = TF.crop(img_exptC, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            a = np.random.uniform(0.6,1.4)
            img_input = TF_x.adjust_brightness(img_input,a)

            #a = np.random.uniform(0.8,1.2)
            #img_input = TF_x.adjust_saturation(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)

class ImageDataset_HDRplus_unpaired(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode

        file = open(os.path.join(root,'train.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"middle_480p",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"output_480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"middle_480p",set2_input_files[i][:-1] + ".png"))
            self.set2_expert_files.append(os.path.join(root,"output_480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"middle_480p",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"output_480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            seed = random.randint(1,len(self.set2_expert_files))
            img2 = Image.open(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img2 = img_exptC

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":
            ratio = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio)
            crop_w = round(W*ratio)
            W2,H2 = img2._size
            crop_h = min(crop_h,H2)
            crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF.hflip(img2)

            a = np.random.uniform(0.8,1.2)
            img_input = TF_x.adjust_brightness(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)
