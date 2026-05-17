from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


def main():

    model = YOLO("yolov8s.pt")

    results = model.train(

        # путь к data.yaml
        data=r"C:\kursach\rdd2022\data.yaml",

        # параметры обучения
        epochs=100,
        imgsz=640,
        batch=16,

        # GPU
        device=0 if torch.cuda.is_available() else "cpu",

        # оптимизатор
        optimizer="AdamW",
        lr0=0.001,

        # загрузка данных
        workers=4,
        cache=False,

        # early stopping
        patience=15,

        # сохранение
        project="road_defects",
        name="yolov8s_rdd2022",

        save=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
