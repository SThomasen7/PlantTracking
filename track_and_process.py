#!/home/stasen/Desktop/plantas/env/bin/python
from ultralytics import YOLO
import cv2
from optparse import OptionParser
import os
import pathlib
from openpyxl import Workbook, load_workbook
import numpy as np
import pandas as pd
import operator

parser = OptionParser()

parser.add_option(
    "-m", "--model",
    dest="model",
    help="Path to YOLO model weights",
    metavar="MODEL"
)

parser.add_option(
    "-p", "--multiprocess",
    dest="multiprocess",
    help="Path to directory",
    metavar="DIR"
)

parser.add_option(
    "-w", "--excel-workbook",
    dest="workbook",
    help="Excel spreadsheet to write to"
)

(options, args) = parser.parse_args()

def track_video(model, base_path, data):
    video = os.path.join(base_path, data["video"])
    print(f"Processing video: {video}")
    cap = cv2.VideoCapture(video)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_dir = os.path.join(base_path, 'output')
    os.makedirs(output_dir, exist_ok=True)

    base_file = os.path.basename(video)
    out_path = os.path.join(output_dir, f"{base_file}_tracked.mp4")
    print(out_path)
    out = cv2.VideoWriter(
        out_path,
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

    # save the data:
    data["conteo_automatico"] = total
    data["error_pct"] = (np.abs(data["conteo_humano"]-total)/data["conteo_humano"])*100
    data["densidad"] = (data["conteo_humano"]/((data["longitud"]*1.95))*10000)
    out.release()

    return data

def load_workbook(filename):
    df = pd.read_excel(filename)
    df = df.rename(columns={
        "Nombre de video": "video",
        "Fecha": "fecha",
        "Linea": "linea",
        "Longitud (metros)": "longitud",
        "Conteo humano": "conteo_humano",
        "Conteo Automatico": "conteo_automatico",
        "% Error": "error_pct",
        "Densidad p/hec": "densidad"
    })
    df = df.where(pd.notnull(df), None)
    data = df.to_dict(orient="records")
    return data

def save_workbook(data, filename):
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "video": "Nombre de video",
        "fecha": "Fecha",
        "linea": "Linea",
        "longitud": "Longitud (metros)",
        "conteo_humano": "Conteo humano",
        "conteo_automatico": "Conteo Automatico",
        "error_pct": "% Error",
        "densidad": "Densidad p/hec"
    })
    df.to_excel(filename, index=False)


def generate_html(data):
    max_longitud = max(d["longitud"] for d in data)
    rows = []
    data = sorted(data, key=operator.itemgetter('linea'))
    for d in data:
        width_percent = (d["longitud"] / max_longitud) * 100

        row = f"""
        <div class="row">
            <div class="linea">{d['linea']}</div>
            <div class="bar-container">
                <div class="bar" style="width: {width_percent}%"></div>
            </div>
            <div class="stats">
                conteo: {d['conteo_humano']} | densidad: {d['densidad']}
            </div>
        </div>
        """
        rows.append(row)

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
            }}
            .row {{
                display: flex;
                align-items: center;
                margin: 6px 0;
            }}
            .linea {{
                width: 60px;
                text-align: right;
                margin-right: 10px;
                font-weight: bold;
            }}
            .bar-container {{
                width: 300px;
                height: 20px;
                background: #eee;
                margin-right: 10px;
            }}
            .bar {{
                height: 100%;
                background: green;
            }}
            .stats {{
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        {''.join(rows)}
    </body>
    </html>
    """
    return html
        

if __name__ == "__main__":
    # excel workbook
    excel_workbook = options.workbook
    if excel_workbook is None:
        excel_workbook = "data.xlsx"

    # Get the model name
    if options.model is None:
        model_path = "runs/detect/train2/weights/best.pt"
    else:
        model_path = options.model

    if options.multiprocess is None:
        print("A directory path is required, passed with -p")
        sys.exit(1)

    # get the files and video path
    files = os.listdir(options.multiprocess)
    base_path = options.multiprocess

    # make output directory to write tracked videos to
    os.makedirs(os.path.join(base_path, 'output'), exist_ok=True)

    # load data workbook
    wb_path = os.path.join(base_path, excel_workbook)
    data = load_workbook(wb_path)

    print(f"Model: {model_path}")
    # track the videos and update the 
    for i in range(len(data)):
        # skip those that we've already predicted
        if not np.isnan(data[i]["conteo_automatico"]):
            continue 
        model = YOLO(model_path)
        data[i] = track_video(model, base_path, data[i])
        save_workbook(data, wb_path)
    
    # create the html report
    html = generate_html(data)
    with open(os.path.join(base_path, 'data.html'), 'w') as fptr:
        fptr.write(html)

