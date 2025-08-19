# Библиотеки
import torch
import torchvision
from pathlib import Path
from torchvision.transforms import ToTensor
from torchvision import transforms
import os
from torch import nn
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time
 
# девайм
from torch.cuda import is_available
device = "cuda" if is_available() else "cpu"

# Трансофрмации для изображений
transfrom_first_nn = torchvision.transforms.Compose([
    transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transfrom_second_nn = torchvision.transforms.Compose([
    transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Списки классов
classes_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
food_or_not = ['Food', 'Not Food' ]

# Модели
first_model = torch.load("models/first_model_effnet_b0.pth", weights_only=False).to(device)
second_model = torch.load("models/second_model_TiV_16_b.pth", weights_only=False).to(device)

# Основной блок кода с получением изображения и выводом предиктов ( нажать q чтобы закрыть)
cap = cv2.VideoCapture(0) 
t0 = time.time()

while True:
    ret, frame = cap.read()
    
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    transformed_image_first = transfrom_first_nn(pil_image).unsqueeze(dim=0).to(device)

    first_model.eval()
    with torch.inference_mode():
        proba = first_model(transformed_image_first)
        logit = torch.round(torch.sigmoid(proba)).type(torch.int16)
        pred = food_or_not[logit]


        if logit == 0:
            second_model.eval()
            with torch.inference_mode():

                transformed_image_second = transfrom_second_nn(pil_image).unsqueeze(dim=0).to(device)
                second_proba = second_model(transformed_image_second)

                zxc = torch.topk(torch.round(torch.softmax(second_proba, dim=1), decimals=3)*100, k=3)
                topk_first = f"{classes_names[zxc.indices[0][0]]} - {zxc.values[0][0]:.2f}%"
                topk_second = f"{classes_names[zxc.indices[0][1]]} - {zxc.values[0][1]:.2f}%"
                topk_third = f"{classes_names[zxc.indices[0][2]]} - {zxc.values[0][2]:.2f}%"


                text_pred = f"Type: {pred} Pred: {classes_names[torch.argmax(second_proba)]}"
                cv2.putText(frame, text_pred, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, topk_first, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, topk_second, (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, topk_third, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        FPS = 1/(time.time() - t0)
        t0= time.time()
        FPS_text = f"FPS - {FPS:.2f}"
        #cv2.putText(frame, str(logit), (5, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, FPS_text, (5, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    

    if cv2.waitKey(1) == ord('q'): 
            break
    

cap.release()
cv2.destroyAllWindows()