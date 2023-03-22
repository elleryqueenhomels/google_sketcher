import argparse
import os

from train import train_model
from utils import download_data

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="data", help="The training data directory")
parser.add_argument("--model_dir", default="model", help="The output model directory")
parser.add_argument("--model_file", default="keras_model.h5", help="The trained model file name")
parser.add_argument("--labels_file", default="categories.txt", help="The labels file name")
parser.add_argument("--classes_file", default="class_names.txt", help="The classes file name")

args = parser.parse_args()

DATA_DIR = parser.data_dir
MODEL_DIR = parser.model_dir
MODEL_FILE = parser.model_file
LABELS_FILE = parser.labels_file
CLASSES_FILE = parser.classes_file
BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'


def main():
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    download_data(data_dir=DATA_DIR, labels_file=LABELS_FILE, base_url=BASE_URL)

    model_save_path = os.path.join(MODEL_DIR, MODEL_FILE)
    classes_file_path = os.path.join(MODEL_DIR, CLASSES_FILE)
    model = train_model(DATA_DIR, model_save_path, classes_file_path)

    print('>>> All done! Model saved to <%s>' % model_save_path)


if __name__ == '__main__':
    main()

