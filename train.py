from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')

    print("🚀 TRAINING WITH UPGRADED SETTINGS...")
    
    results = model.train(
        data='data.yaml', 
        epochs=100,
        imgsz=640,
        batch=4,
        workers=4,
        device=0,
        patience=15,
        cos_lr=True,
        mixup=0.1,
        close_mosaic=10,
    )
