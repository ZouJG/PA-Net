from data.my_dataset import *



my_data = CocoDataset()

my_data.load_coco("/media/jack/game1/ISAID","train")

my_data.load_image(1)