from ultralytics import YOLO


def export_to_trt(model_path, batch, half=False):
    model = YOLO(model_path)
    try:
        # Export the model to TensorRT format
        model.export(format="engine", 
                    dynamic=True,
                    imgsz=640,
                    half=half,
                    batch=batch) 
    except Exception as e:
        print(e)
    

if __name__ == '__main__':
    export_to_trt("/home/Документы/lct_2024/lct_2024/weights/best.pt", batch=8)