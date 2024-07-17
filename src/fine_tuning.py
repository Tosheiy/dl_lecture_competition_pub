from torch import nn, optim
import clip
import torch
from torch.utils.data import DataLoader
from src.datasets import CustomDataset
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm


def class_name(name):
     if '/' in name:
          c_name = name.split('/')[0]
     else:
          c_name_list = name.split('_')
          c_name_list.pop()
          c_name = ''
          for i, c_name_i in enumerate(c_name_list):
               if i > 0:
                    c_name = c_name + "_" + c_name_i
               else:
                    c_name = c_name_i
     return c_name

def img_path(name):
     if '/' in name:
          return name
     else:
          c_name_list = name.split('_')
          c_name_list.pop()
          c_name = ''
          for i, c_name_i in enumerate(c_name_list):
               if i > 0:
                    c_name = c_name + "_" + c_name_i
               else:
                    c_name = c_name_i
          c_name = c_name + '/' + name
          return c_name

def fine_tuning():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    dir_path = ".\\data\\images\\Images"
    texts_train = open("./class_y.txt", "r").read().split('\n')

    image_features = torch.empty((1854)).to(device)
    for class_name in tqdm(texts_train):
        files = os.listdir(os.path.join(dir_path, class_name))
        image_features_cls = torch.empty((0, 512)).to(device)
        for img in files: 
            with torch.no_grad():
                image = preprocess(Image.open(os.path.join(dir_path, class_name, img))).unsqueeze(0).to(device)
                image_feature = model.encode_image(image)
            image_features_cls = torch.cat((image_features_cls, image_feature), dim=0)
        image_features = torch.cat((image_features, torch.mean(image_features_cls, dim=0)), dim=0)
    torch.save(image_features, 'mean_image_features.pt')

    image_features = torch.empty((0, 512)).to(device)
    texts_train = open("./data/train_image_paths.txt", "r").read().split('\n')
    for file in tqdm(texts_train):
        class_name = img_path(file)
        files = os.path.join(dir_path, class_name)
        with torch.no_grad():
            image = preprocess(Image.open(files)).unsqueeze(0).to(device)
            image_feature = model.encode_image(image)
        image_features = torch.cat((image_features, image_feature), dim=0)
    torch.save(image_features, 'x_train_image_features.pt')

    image_features = torch.empty((0, 512)).to(device)
    texts_train = open("./data/val_image_paths.txt", "r").read().split('\n')
    for file in tqdm(texts_train):
        class_name = img_path(file)
        files = os.path.join(dir_path, class_name)
        with torch.no_grad():
            image = preprocess(Image.open(files)).unsqueeze(0).to(device)
            image_feature = model.encode_image(image)
        image_features = torch.cat((image_features, image_feature), dim=0)
    torch.save(image_features, 'x_val_image_features.pt')