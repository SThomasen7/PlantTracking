#!/home/stasen/Desktop/plantas/env/bin/python
from ultralytics import YOLO
import cv2
from optparse import OptionParser
import os

parser = OptionParser()

parser.add_option(
    "-m", "--model",
    dest="model",
    help="Path to YOLO model weights",
    metavar="FILE"
)

parser.add_option(
    "-v", "--video",
    dest="video",
    help="Path to input video",
    metavar="FILE"
)

(options, args) = parser.parse_args()

def track_video(model, video):
    cap = cv2.VideoCapture(video)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs("output", exist_ok=True)

    out = cv2.VideoWriter(
        f"{video[:-4]}_tracked.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    results = model.track(
        source=video,
        conf=0.4,
        iou=0.5,
        tracker="bytetrack.yaml",
        stream=True
    )

    ## Iterate over the frames and get the count of unique ids
    seen_ids = dict()
    for r in results:
        frame = r.plot()

        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy()
            for i in ids:
                if int(i) not in seen_ids.keys():
                    seen_ids[int(i)] = 0
                seen_ids[int(i)] += 1

        total = 0
        for key, val in seen_ids.items():
            if val > 5:
                total += 1

        cv2.putText(
            frame,
            f"Conteo: {total}",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (100, 255, 100),
            2
        )

        out.write(frame)
    out.release()


if __name__ == "__main__":

    if options.model is None:
        model_path = "runs/detect/train2/weights/best.pt"
    else:
        model_path = options.model

    if options.video is None:
        video_path = "/home/stasen/Desktop/plantas/videos/linea_841.mp4"
    else:
        video_path = options.video

    print(f"Detecting video: {video_path}")
    print(f"Model: {model_path}")
    
    model = YOLO(model_path)
    track_video(model, video_path)

