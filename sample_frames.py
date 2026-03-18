#!/bin/python3
import cv2
import numpy as np
import os
import sys
from optparse import OptionParser

parser = OptionParser()

parser.add_option(
    "-v", "--video",
    dest="video",
    help="Directory of videos, or a video name",
)

parser.add_option(
    "-o", "--output-dir",
    dest="output",
    help="Directory to write out sampled frames."
)

parser.add_option(
    "-n", "--num_samples",
    dest="num_samples",
    help="Number of frames to take from the video."
)

(options, args) = parser.parse_args()

def process_video(filename, out_dir, num_samples):
    os.makedirs(out_dir, exist_ok=True)

    # video id = filename without extension
    video_id = os.path.splitext(os.path.basename(filename))[0]

    cap = cv2.VideoCapture(filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, int(num_samples), dtype=int)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        out_path = os.path.join(out_dir, f"{video_id}_frame_{i:03d}.png")
        cv2.imwrite(out_path, frame)

    cap.release()
    print(f"Saved {num_samples} frames from {filename} to {out_dir}")


if __name__ == "__main__":
    # process the command line arguments.
    if options.video is None:
        print("Specify the directory to read the videos from using --video")
        sys.exit(1)

    # determine the output directory
    output = options.output
    if options.output is None:
        print("No output directory specified, using 'output'")
        output = "output"
    os.makedirs(output, exist_ok=True)

    # get the number of frames
    n = options.num_samples
    if options.num_samples is None:
        print("Number of frames not specified, sampling 100")
        n = 100

    # process the video
    if options.video.endswith("mp4"):
        process_video(options.video, output, n)
    else:
        videos = os.listdir(options.video)
        for video in videos:
            if not video.endswith("mp4"):
                continue
            process_video(os.path.join(options.video, video),
                          output,
                          n
                    )

