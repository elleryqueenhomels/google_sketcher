import os

from train import train_model
from utils import download_data


DATA_DIR = 'data'
MODEL_DIR = 'model'
MODEL_FILE = 'keras_model.h5'
LABELS_FILE = 'categories.txt'
CLASSES_FILE = 'class_names.txt'
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

