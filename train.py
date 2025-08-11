from ultralytics import YOLO
import torch

def train_yolo_advanced():
    # Initialize model
    model = YOLO('yolov8n.pt')

    # Custom training settings
    config = {
        'data': 'data.yaml',
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'patience': 50,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'project': 'runs/train',
        'name': 'exp',
        'pretrained': True,
        'optimizer': 'Adam',
        'lr0': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'save': True,
        'save_period': 10,  # save checkpoint every 10 epochs
        'cache': False,
        'val': True,        # validate during training
        'plots': True,      # plot training results
        'rect': False,      # rectangular training
        'resume': False,    # resume training from last checkpoint
        'amp': True,       # Automatic Mixed Precision
        'fraction': 1.0,   # dataset fraction to train on
        'cos_lr': True,    # cosine learning rate scheduler
        'label_smoothing': 0.0,  # label smoothing
        'overlap_mask': True,    # masks should overlap during training
        'mask_ratio': 4,        # mask downsample ratio
        'dropout': 0.0,         # use dropout regularization
        'single_cls': False,    # train as single-class dataset
    }

    # Start training
    try:
        results = model.train(**config)
        
        # Print training results
        print("\nTraining Results:")
        print(f"Best Epoch: {results.best_epoch}")
        print(f"Best mAP50: {results.maps[50]}")
        print(f"Best mAP50-95: {results.maps[0]}")
        
        # Validate the model
        print("\nValidating final model...")
        metrics = model.val()
        print("Validation Results:")
        print(f"mAP50: {metrics.maps[50]}")
        print(f"mAP50-95: {metrics.maps[0]}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: Training on CPU (this will be slow)")
    
    train_yolo_advanced()