# PlantTracking

## Training the model
### Sample frames from a video

To sample frames from a video for manual annotation, run sample_frames.py

```
Usage: sample_frames.py [options]

Options:
  -h, --help            show this help message and exit
  -v VIDEO, --video=VIDEO
                        Directory of videos, or a video name
  -o OUTPUT, --output-dir=OUTPUT
                        Directory to write out sampled frames.
  -n NUM_SAMPLES, --num_samples=NUM_SAMPLES
                        Number of frames to take from the video.
```

Pass a video to -v to process a video, or a directory to process all of the videos within the directory. Only videos of type .mp4 are considered.


### Labeling the images
I recommend labeling the images with LabelImg or another third party.

### Organize the image directory for YOLO.
The directory must have the following structure.

dataset/
 | data.yaml - information on the dataset
 | images 
 |   | train
 |   | test
 |   \ val
 \ labels 
     | train
     | test
     \ val

You can build this using build\_data.py.

```
$ python build_data.py --help
Usage: build_data.py [options]

Options:
  -h, --help            show this help message and exit
  -i IMAGE_DIR, --image-dir=IMAGE_DIR
                        Directory of images to train on
  -l LABEL_DIR, --label-dir=LABEL_DIR
                        Directory of YOLO style label files.
  -d DATA_DIR, --dataset=DATA_DIR
                        Directory to store the split data for YOLO to train on.
  -s SEED, --seed=SEED  Random seed.
  -t TEST, --test-split=TEST
                        Test Split, recommended 0.2
  -v VAL, --val-split=VAL
                        Validation split, recommended 0.2
```

Warning, if the datadir exists, you will be prompted to delete it. Be careful not to delete something important.

### Fine tune the model
Run the model, specifying where your dataset yaml is stored.
$ yolo train data=dataset/data.yaml model=yolo26n.pt augment=True
