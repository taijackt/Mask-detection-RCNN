# import all the tools we need
import urllib
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import os 
from PIL import Image
import random
import time
import requests
from datetime import datetime
import yaml

# local
from tools import read_annot, draw_boxes
from dataset import image_dataset, formatting
from train import train

def print_log(message):
    t = datetime.now().strftime("%d/%m/%y - %H:%M:%S")
    print(f"({t}): {message}")
    # unfinished.

def read_configs(yamlPath="./configs.yaml"):
    with open(yamlPath, 'r') as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    return configs

def get_model(num_classes=3, isPretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=isPretrained)

    # Change the predictor 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

def main():

    # Read configs from yaml file
    configs = read_configs(yamlPath="./configs.yaml");

    dir_path = configs["IMAGE_DIR_PATH"]# Path of image folder
    xml_path = configs["XML_DIR_PATH"] # path of xml folder
    
    file_list = os.listdir(dir_path)
    print_log(f"There are total {len(file_list)} images.")

    # get dataset, get dataloader
    mask_dataset = image_dataset(file_list, dir_path, xml_path)
    mask_loader = DataLoader(mask_dataset, batch_size=int(configs["BATCH_SIZE"]), shuffle=True, num_worker=2, collate_fn=formatting)
    print_log("DataLoader is ready.")

    # Set the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Get the model ready
    model = get_model(num_classes=int(configs["NUM_CLASSES"]), isPretrained=True)
    model = model.to(device)
    print("Model is ready.")

    # Set the optimizer
    params = [p for p in model.parameters() if p.required_grad]
    optimizer = torch.optim.SGD(params, 
                                lr=configs["LEARNING_RATE"],
                                momentum=configs["MOMENTUM"], 
                                weight_decay=configs["WEIGHT_DECAY"])
    
    # epochs
    EPOCHS = int(configs["EPOCHS"])

    # Start training
    train(
        model=model, 
        num_epochs=EPOCHS, 
        loader=mask_loader, 
        optimizer=optimizer, 
        device=device
    )
    print_log("Finished training.")


    # Save the trained model
    torch.save(model.state_dict(), configs["SAVE_PATH"])
    print_log(f"Trained model saved. Path: {configs['SAVE_PATH']}")


if __name__ =="__main__":
    #main()
