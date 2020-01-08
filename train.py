import argparse
from data.my_dataset import *
from model.panet import *


log_dir = DEFAULT_LOGS_DIR

config = CocoConfig()
config.display()

# Create model
if True:
    model = MaskRCNN(mode="training", config=config,
                              model_dir=log_dir)
else:
    
    model = MaskRCNN(mode="inference", config=config,
                              model_dir=log_dir)


model_path = model.get_imagenet_weights()


# Load weights
print("Loading weights ", model_path)
model.load_weights(model_path, by_name=True)

# Train or evaluate
if True:
    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = CocoDataset()
    dataset_train.load_coco("/media/jack/game1/ISAID_new", "train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = CocoDataset()
    val_type = "val"
    dataset_val.load_coco("/media/jack/game1/ISAID_new", val_type)
    dataset_val.prepare()
    
    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)
    
    # *** This training schedule is an example. Update to your needs ***
    
    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)
    
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation)
    
    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)
