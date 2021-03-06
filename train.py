import os

from data.prepareData import CocoDataset
from models.CenterNet_52 import CenterNet52
from config import COCOConfig


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config = COCOConfig()

    model = CenterNet52(config=config,
                              weight_dir=config.MODLE_DIR)

    dataset_train = CocoDataset()
    dataset_train.load_coco(config.DATA_DIR, "val", auto_download=True)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()
    dataset_val.load_coco(config.DATA_DIR, "val", auto_download=True)
    dataset_val.prepare()
    print("Training network")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LR_VALS[0],
                epochs = config.EPOCHS,
                augment=True)
    model.save("fine.h5")