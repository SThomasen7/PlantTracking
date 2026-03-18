#!/bin/python3
from sklearn.model_selection import train_test_split
from optparse import OptionParser
import os
import sys
import shutil

RANDOM_SEED=42

parser = OptionParser()
parser.add_option(
    "-i", "--image-dir",
    dest="image_dir",
    help="Directory of images to train on",
    default="images"
)

parser.add_option(
    "-l", "--label-dir",
    dest="label_dir",
    help="Directory of YOLO style label files.",
    default="labels"
)

parser.add_option(
    "-d", "--dataset",
    dest="data_dir",
    help="Directory to store the split data for YOLO to train on.",
    default="dataset"
)

parser.add_option(
    "-s", "--seed",
    dest="seed",
    help="Random seed.",
    default=42,
    type="int"
)

parser.add_option(
    "-t", "--test-split",
    dest="test",
    help="Test Split, recommended 0.2",
    default=0.2,
    type="float"
)

parser.add_option(
    "-v", "--val-split",
    dest="val",
    help="Validation split, recommended 0.2",
    default=0.2,
    type="float"
)

(options, args) = parser.parse_args()

def split_dataset(images_dir, labels_dir, output_dir, test_size=0.2, 
                    val_size=0.2, seed=42):
    # get all of the images
    images = [f for f in os.listdir(images_dir) if  \
                (f.endswith('.jpg') or f.endswith('.png'))]

    # make the train test validation split.
    train_images, test_images = train_test_split(images, test_size=test_size, 
                                                 random_state=seed)
    train_images, val_images = train_test_split(train_images, test_size=val_size, 
                                                 random_state=seed)

    if os.path.exists(output_dir):
        print("The output dir already exists. Would you like to delete it")
        print("y-delete, n-abort")
        choice = input('').split(" ")[0].lower()
        print(choice)
        if choice == 'y':
            print("Deleting!")
            shutil.rmtree(output_dir)
        else:
            print("Aborting...")
            sys.exit(1)

    os.makedirs(output_dir)

    for subset, subset_images in [('train', train_images), \
                                  ('val', val_images), ('test', test_images)]:
        image_path = os.path.join(output_dir, 'images', f'{subset}')
        label_path = os.path.join(output_dir, 'labels', f'{subset}')
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)
        # for each image copy over.
        for image in subset_images:
            shutil.copy(os.path.join(images_dir, image), os.path.join(image_path, image))
            label_file = image.replace('.jpg', '.txt')
            label_file = label_file.replace('.png', '.txt')

            # If there is no corresponding text file, write a blank text file,
            # assume it is labeled, but there are no plants.
            if os.path.exists(os.path.join(labels_dir, label_file)):
                with open(os.path.join(labels_dir, label_file), 'r') as fptr:
                    lines = list(fptr)

                with open(os.path.join(label_path, label_file), "w") as fptr:
                    for line in lines:
                        fptr.write(
                                '0 '+' '.join(line.strip().split(" ")[1:])
                shutil.copy(os.path.join(labels_dir, label_file),
                            os.path.join(label_path, label_file))
            else:
                open(os.path.join(label_path, label_file), "w").close()

    # write the yaml file:
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as fptr:
        fptr.write(
f"""names:
  0: planta
path: dataset
train: {os.path.join('images', 'train')}
val: {os.path.join('images', 'val')}
test: {os.path.join('images', 'test')}
nc: 1 
""")

if __name__ == "__main__":
    image_dir = options.image_dir
    label_dir = options.label_dir
    data_dir = options.data_dir
    seed = options.seed
    test_size = options.test
    val_size = options.val

    print(f"""
Images dir: {image_dir}
Label dir: {label_dir}
Dataset dir: {data_dir}
seed: {seed}
test_size: {test_size}
val_size: {val_size}
""")

    split_dataset(image_dir, label_dir, data_dir, test_size=test_size, 
                  val_size=val_size, seed=seed)


