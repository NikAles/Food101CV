# Food Detection & Recognition

Программа для распознавания еды в реальном времени с веб-камеры.  
Использует две нейросети:  

1. **Food or Not** — определяет, есть ли на кадре еда.  
2. **Food Classifier** — если еда есть, классифицирует, какой это тип блюда и выводит топ-3 предсказания с вероятностями.

---

## Как это работает

1. С помощью OpenCV захватывается кадр с веб-камеры.  
2. Кадр проходит через первую модель: **Food or Not**.  
3. Если первая модель определяет, что еда есть, кадр передаётся во вторую модель: **Food Classifier**.  
4. На экран выводится окно с топ-3 предсказаниями и их вероятностями.  

---

## Используемые модели и датасеты

- **Первая модель (Food/Not Food)**
  - Архитектура: EfficientNet-B0
  - Датасет: Kaggle Food or Not
  - Точность: ~94%

- **Вторая модель (Food Classifier)**
  - Архитектура: ViT-B16
  - Датасет: FOOD101
  - Точность: ~75–78%

---

## Преобразования изображений

Для каждой модели используется своя трансформация кадра:

```python
from torchvision import transforms

# Первая модель
transform_first_nn = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Вторая модель
transform_second_nn = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
