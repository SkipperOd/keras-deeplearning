import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

Data_DIR = "./PetImages/"
Categories = []


for filename in os.listdir(Data_DIR):
    Categories.append(filename)
    category_path = os.path.join(Data_DIR, filename)
    for img_name in os.listdir(category_path):
        path = os.path.join(category_path, img_name)
        print(path)
        img = cv2.imread(path)
        cv2.imshow("data", img)
        cv2.waitKey(0)
