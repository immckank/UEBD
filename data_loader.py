import torch
import torch.nn.functional as F
import random
import numpy as np
import logging

import torchattacks

from pathlib import Path
from itertools import combinations, permutations
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image

from models.resnet import *

cl = ['4ColoredBlockTrigger', 'fiveBlockTrigger', 'WaNetTrigger']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
benign_model = torchattacks.resnet18()
ckpt = Path('./save_models/cifar_resnet_e8_a2_s10.pth')
ckpt = torch.load(ckpt)
benign_model.load_state_dict(ckpt)
benign_model.to(device)
benign_model.eval()
atk = torchattacks.PGD(benign_model, eps=16/255, alpha=2/255, steps=10)

def get_CL_train_loader(opt):
    print('get_CL_train_loader')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')
    
    train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    
    return train_loader
    
def get_BD_train_loader(opt):
    print('get_BD_train_loader')
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')
    
    train_data = DatasetBD(opt, full_dataset=trainset, inject_portion=1.0, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    
    return train_loader    
    
def get_CL_test_loader(opt):
    print('get_CL_test_loader')
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if (opt.dataset == 'CIFAR10'):
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
    else:
        raise Exception('Invalid dataset')
    
    test_data = DatasetCL(opt, full_dataset=testset, transform=tf_test)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    
    return test_loader

class SimpleDataset(Dataset):
    '''
    construct with PIL.Image
    '''
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)
    
    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        # PIL.Image to np.array
        image = np.array(image)
        image = image / 255
        # np.array to torch.tensor
        image = torch.tensor(image, dtype=torch.float)
        # np:image [H W C]
        # tensor:image [C H W]
        image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def __len__(self):
        return self.dataLen

class DatasetCL(Dataset):
    '''
    clean dataset
    '''
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)
        self.logger = logging.getLogger(__name__)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen
    
class DatasetBD(Dataset):
    '''
    Dataset with backdoor
    '''
    def __init__(self, opt, full_dataset, 
                inject_portion, transform=None, mode="train", 
                device=torch.device("cuda"), distance=1):
        # self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w, opt.trig_h, opt.trigger_type, opt.target_type)
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.class_dict = self.generateClassDict(full_dataset)
        if opt.trigger_type in cl:
            self.dataset = self.addAP(full_dataset)
        self.dataset = self.addUEBD(self.dataset, inject_portion=1.0, 
                                    mode=mode, distance=distance, 
                                    trig_w=0, trig_h=0, trigger_type=opt.trigger_type, opt=opt)
        self.transform = transform 
        
    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.dataset)
    
    def generateClassDict(self, dataset):
        '''
        generate a dict of class: classidx
        '''
        class_dict = {}
        classidx = 0
        for i in range(len(dataset)):
            label = dataset[i][1]
            if label not in class_dict:
                class_dict[label] = classidx
                classidx += 1
        # print(f'class_dict len: {len(class_dict)}')
        return class_dict
    
    def addUEBD(self, dataset, inject_portion, 
                mode, distance, 
                trig_w, trig_h, trigger_type, opt):
        '''
        Simple implementation of UEBD
            for train
            add trigger with different color for different classes             
        '''
        if opt.defaultTrigger == True:
            self.trigger_list = [(0, 6, 7, 8), (4, 5, 6, 8), (2, 5, 6, 7), (0, 5, 6, 7), (1, 2, 3, 6), (2, 4, 5, 8), (3, 4, 6, 7), (0, 1, 2, 3), (0, 4, 5, 8), (0, 4, 7, 8)]
            self.color_list = [(1, 3, 0, 2), (0, 3, 1, 2), (1, 0, 2, 3), (0, 2, 1, 3), (2, 1, 0, 3), (2, 3, 0, 1), (3, 2, 0, 1), (2, 0, 1, 3), (2, 1, 3, 0), (3, 1, 2, 0)]
        else:
            self.trigger_list, self.color_list = self.triggerGenerator(trigger_type, trig_w, trig_h, len(self.class_dict))
        dataset_ = []
        # to get size of image
        image = dataset[0][0]
        label = dataset[0][1]
        image = np.array(image)
        width, height = image.shape[0], image.shape[1]
        
        for item in dataset:
            image = self.addTrigger(item, width, height, trigger_type)
            label = item[1]
            dataset_.append((image, label))
        
        return dataset_
    
    def triggerGenerator(self, trigger_type, trig_w, trig_h, class_num):
        '''
        generate a list of trigger with (different color)
        '''
        trigger_list = []
        color_list = []
        if trigger_type == "badnet":
            # badnet trigger: 3x3 square: 9 items
            # list of trigger: class_num * 9
            # 10 as an example
            # L1 = [0,0,0,0,0,0,0,0,0,0]
            for i in range(class_num):
                trigger = [255, 0, 255,
                           0, 255, 0,
                           255, 0, 0]
                trigger_list.append(trigger)
            print(f"trigger_list: {trigger_list}")
            self.logger.info(f"trigger_list: {trigger_list}")
        elif trigger_type == "demo":
            # demo trigger: 3x3 square: 9 items
            # list of trigger: class_num * 9
            # 10 as an example
            # L1 = [0,0,0,0,0,0,0,0,0,0]
            for i in range(class_num):
                trigger = [0, 0, 0,
                           0, 0, 0,
                           0, 0, 0]
                if i != 9:
                    trigger[i] = 255
                trigger_list.append(trigger)
                
            print(f"trigger_list: {trigger_list}")
            self.logger.info(f"trigger_list: {trigger_list}")
        elif trigger_type == "random_demo":
            # demo trigger: 3x3 square: 9 items
            # list of trigger: class_num * 9
            # 10 as an example
            L1 = random.sample(range(0, 255), 9)
            for i in range(class_num):
                trigger = [255, 255, 255,
                            255, 255, 255,
                            255, 255, 255]
                if i != 9:
                    trigger[i] = L1[i]
                trigger_list.append(trigger)
                
            print(f"trigger_list: {trigger_list}") 
            self.logger.info(f"trigger_list: {trigger_list}")
        elif trigger_type == "watermark":
            # watermarked block 5*5
            # markinside 3*3
            # random color 1*1
            color_l = range(0, 256)
            all_comb = combinations(color_l, 3)
            all_color = list(all_comb)
            random.shuffle(all_color)
            trigger_list = random.sample(all_color, class_num)
            print(f"trigger_list: {trigger_list}")
            self.logger.info(f"trigger_list: {trigger_list}")
        elif trigger_type == "samplewiseReTrigger":
            print("different trigger for different sample, samplewiseReTrigger")
            self.logger.info("different trigger for different sample, samplewiseReTrigger")
        elif trigger_type == "samplewiseLiTrigger":
            print("different trigger for different sample, samplewiseLiTrigger")
            self.logger.info("different trigger for different sample, samplewiseLiTrigger")
        elif trigger_type == "4ColoredBlockTrigger":
            blockid_l = range(0, 9)
            all_comb = combinations(blockid_l, 4)
            all_shape = list(all_comb)
            random.shuffle(all_shape)
            # print(f"all_shape: {all_shape}")
            # print(f"len(all_shape): {len(all_shape)}")
            trigger_list = random.sample(all_shape, class_num)
            color_list = range(0, 4)
            all_perm = permutations(color_list, 4)
            all_color = list(all_perm)
            random.shuffle(all_color)
            color_list = random.sample(all_color, class_num)
            self.logger.info(f"trigger_list: {trigger_list}")
            self.logger.info(f"color_list: {color_list}")
        elif trigger_type == "fiveBlockTrigger":
            blockid_l = range(0, 9)
            all_comb = combinations(blockid_l, 5)
            all_shape = list(all_comb)
            random.shuffle(all_shape)
            # print(f"all_shape: {all_shape}")
            # print(f"len(all_shape): {len(all_shape)}")
            trigger_list = random.sample(all_shape, class_num)
            for i in range(len(trigger_list)):
                color_list.append(random.sample(trigger_list[i], 1)) 
            print(f"trigger_list: {trigger_list}")
            print(f"color_list: {color_list}")
            self.logger.info(f"trigger_list: {trigger_list}")
        elif trigger_type == "WaNetTrigger":
            # WaNet trigger full image
            # To gengerate one type of trigger
            # get noise_grid and identity_grid
            for i in range(class_num):
                # noise_grid 
                
                # k = 4 as default
                ins = torch.rand(1, 2, 4, 4) * 2 - 1
                ins = ins / torch.mean(torch.abs(ins))
                # trigger hight as 32/ size
                noise_grid = (
                    F.upsample(ins, size=32, mode="bicubic", align_corners=True)
                    .permute(0, 2, 3, 1)
                )
                # print(f"noise_grid: {noise_grid}")
                
                # identity_grid
                # trigger hight as 32/ step
                array1d = torch.linspace(-1, 1, steps=32)
                x, y = torch.meshgrid(array1d, array1d)
                identity_grid = torch.stack((y, x), 2)[None, ...]
                # print(f"identity_grid: {identity_grid}")
                trigger = [noise_grid, identity_grid]
                trigger_list.append(trigger)
            
            self.logger.info("WaNet trigger full image")
            self.logger.info(f"trigger_list: {trigger_list}")
        else:
            raise NotImplementedError
        
        return trigger_list, color_list
          
    def addTrigger(self, data, width, height, trigger_type):
        '''
        add specific trigger for one data
        '''
        if trigger_type == "badnet":
            # badnet trigger: 3x3 square: 9 items
            # list of trigger: class_num * 9
            # 10 as an example
            image = self.demoTrigger(data, width, height)
        elif trigger_type == "demo":
            # demo trigger: 3x3 square: 9 items
            # list of trigger: class_num * 9
            # 10 as an example
            image = self.demoTrigger(data, width, height)
        elif trigger_type == "random_demo":
            # demo trigger: 3x3 square: 9 items
            # list of trigger: class_num * 9
            # 10 as an example
            image = self.demoTrigger(data, width, height)
        elif trigger_type == "watermark":
            image = self.watermarkTrigger(data, width, height)
        elif trigger_type == "samplewiseReTrigger":
            image = self.samplewiseReTrigger(data, width, height)
        elif trigger_type == "samplewiseLiTrigger":
            image = self.samplewiseLiTrigger(data, width, height)
        elif trigger_type == "4ColoredBlockTrigger":
            image = self.fourColoredBlockTrigger(data, width, height)
        elif trigger_type == "fiveBlockTrigger":
            image = self.fiveBlockTrigger(data, width, height)
        elif trigger_type == "WaNetTrigger":
            image = self.WaNetTrigger(data, width, height)
        else:
            raise NotImplementedError
        
        return image
            
    def demoTrigger(self, data, width, height):
        label = data[1]
        # print(f"trigger: {self.trigger_list[self.class_dict[label]]}")
        image = np.array(data[0])
        trigger = self.trigger_list[self.class_dict[label]]
        image[width-1][height-1] = trigger[0]
        image[width-1][height-2] = trigger[1]
        image[width-1][height-3] = trigger[2]
        
        image[width-2][height-1] = trigger[3]
        image[width-2][height-2] = trigger[4]
        image[width-2][height-3] = trigger[5]
        
        image[width-3][height-1] = trigger[6]
        image[width-3][height-2] = trigger[7]
        image[width-3][height-3] = trigger[8]
        
        return image
    
    def watermarkTrigger(self, data, width, height):
        label = data[1]
        image = np.array(data[0])
        trigger = self.trigger_list[self.class_dict[label]]
        r_trigger = [255-trigger[0], 255-trigger[1], 255-trigger[2]]
        image[width-2][height-2] = trigger
        image[width-3][height-3] = [255, 0, 0]
        image[width-1][height-1] = [255, 0, 0]
        image[width-1][height-3] = [255, 0, 0]
        image[width-3][height-1] = [255, 0, 0]
        return image
    
    def calsamplewiseReTrigger(self, image, width, height):
        l = []
        l.append(image[width-1][height-2])
        l.append(image[width-2][height-1])
        l.append(image[width-2][height-3])
        l.append(image[width-3][height-2])
        # average color
        color = [0, 0, 0]
        for block in l:
            color[0] += block[0]
            color[1] += block[1]
            color[2] += block[2]
        color[0] = color[0] / 4
        color[1] = color[1] / 4
        color[2] = color[2] / 4
        print(f"color: {color}")
        return color
    
    def samplewiseReTrigger(self, data, width, height):
        label = data[1]
        # print(f"trigger: {self.trigger_list[self.class_dict[label]]}")
        image = np.array(data[0])
        trigger = self.calsamplewiseReTrigger(image, width, height)
        r_trigger = [255-trigger[0], 255-trigger[1], 255-trigger[2]]
        image[width-2][height-2] = r_trigger
        image[width-3][height-3] = [255, 0, 0]
        image[width-1][height-1] = [255, 0, 0]
        image[width-1][height-3] = [255, 0, 0]
        image[width-3][height-1] = [255, 0, 0]
        return image
    
    def samplewiseLiTrigger(self, data, width, height):
        label = data[1]
        # print(f"trigger: {self.trigger_list[self.class_dict[label]]}")
        image = np.array(data[0])
        trigger = image[width-2][height-2]
        trigger = trigger.tolist()
        for i in range(len(trigger)):
            trigger[i] = max(255, 100+255-trigger[i])
        # image[width-2][height-2] = trigger
        image[width-3][height-3] = [255, 0, 0]
        image[width-1][height-1] = [255, 0, 0]
        image[width-1][height-3] = [255, 0, 0]
        image[width-3][height-1] = [255, 0, 0]
        return image
   
    def fourColoredBlockTrigger(self, data, width, height):
        label = data[1]
        label = int(label)
        image = np.array(data[0])
        trigger = self.trigger_list[self.class_dict[label]]
        colors = self.color_list[self.class_dict[label]]
        for i in range(len(trigger)):
            blockid = trigger[i]
            colorid = colors[i]
            color = [0, 0, 0]
            if colorid == 0:
                # red
                color = [255, 0, 0]
            elif colorid == 1:
                # yellow
                color = [255, 255, 0]
            elif colorid == 2:
                # green
                color = [0, 255, 0]
            elif colorid == 3:
                # blue
                color = [0, 0, 255]
            if blockid == 0:
                image[width-3][height-3] = color
            elif blockid == 1:
                image[width-3][height-2] = color
            elif blockid == 2:
                image[width-3][height-1] = color
            elif blockid == 3:
                image[width-2][height-3] = color
            elif blockid == 4:
                image[width-2][height-2] = color
            elif blockid == 5:
                image[width-2][height-1] = color
            elif blockid == 6:
                image[width-1][height-3] = color
            elif blockid == 7:
                image[width-1][height-2] = color
            elif blockid == 8:
                image[width-1][height-1] = color
        return image
        
    def fiveBlockTrigger(self, data, width, height):
        label = data[1]
        label = int(label)
        # print(f"trigger: {self.trigger_list[self.class_dict[label]]}")
        image = np.array(data[0])
        trigger = self.trigger_list[self.class_dict[label]]
        colorid = self.color_list[self.class_dict[label]]
        for i in range(len(trigger)):
            blockid = trigger[i]
            color = [0, 0, 0]
            #if blockid == colorid:
            #    color = [0, 0, 0]
            if blockid%2 == 1:
                color = [255, 255, 255]
            if blockid == 0:
                image[width-3][height-3] = color
            elif blockid == 1:
                image[width-3][height-2] = color
            elif blockid == 2:
                image[width-3][height-1] = color
            elif blockid == 3:
                image[width-2][height-3] = color
            elif blockid == 4:
                image[width-2][height-2] = color
            elif blockid == 5:
                image[width-2][height-1] = color
            elif blockid == 6:
                image[width-1][height-3] = color
            elif blockid == 7:
                image[width-1][height-2] = color
            elif blockid == 8:
                image[width-1][height-1] = color
        return image 
    
    def WaNetTrigger(self, data, width, height):
        label = data[1]
        label = int(label)
        # iamge:np [(N) H W C]
        image = np.array(data[0])
        print(f'size of image: {image.size}')
        # image:torch [(N) C H W]
        image = torch.tensor(image, dtype=float).unsqueeze(0).permute(0, 3, 1, 2)
        print(f'shape of image: {image.shape}')
        trigger = self.trigger_list[self.class_dict[label]]
        noise_grid = trigger[0]
        identity_grid = trigger[1]
        # s = 0.5 / size = 32 / recale = 1
        grid_temps = (identity_grid + 0.5 * noise_grid / 32) * 1
        print(f'shape of grid_temps: {grid_temps.shape}')
        grid_temps = torch.clamp(grid_temps, min=-1, max=1)
        # to device
        grid_temps = grid_temps.to(self.device)
        image = image.float()
        image = image.to(self.device)
        image = F.grid_sample(image, grid_temps, align_corners=True)
        # image:torch [N C H W] to iamge:np [N H W C]
        image = image.permute(0, 2, 3, 1).squeeze(-4)
        image = image.cpu()
        image = np.array(image)
        image = np.uint8(image)
        # print(f'size of image: {image.shape}')
        return image
    
    def addAP(self, dataset):
        # add AdversarialPerturbation to all images in datasets
        # input full dataset of PIL.Image
        # return full dataset of PIL.Image with AP
        _dataset = []
        batch_size = 128
        # get c h w
        c = 3
        h = dataset[0][0].height
        w = dataset[0][0].width
        print(f'c: {c}, h: {h}, w: {w}')
        DS = SimpleDataset(full_dataset=dataset, transform=None)
        dataloader = DataLoader(DS, batch_size=batch_size, shuffle=False)
        # add AP per batch
        for images, labels in dataloader:
            # images: torch.tensor [N C H W]
            images.to(device)
            atk_images = atk(images, labels)
            images = atk_images.cpu()
            images = images * 255
            images = images.byte()
            images = torch.clamp(images, 0, 255)
            images = images.permute(0, 2, 3, 1)
            images = np.array(images)
            images.reshape(len(labels), h, w, c)
            batch = zip(images, labels)
            _dataset.extend(batch)
        return _dataset