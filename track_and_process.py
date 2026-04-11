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

    # get the final mapping
    total = 0
    count_map = list()
    for key, val in seen_ids.items():
        if val > 5:
            total += 1
            count_map.append({"count": total, "id": key})

    count_map_file = os.path.join(output_dir, f"{base_file}_tracked.xlsx")
    df = pd.DataFrame(count_map)
    df.to_excel(count_map_file, index=False)

    # save the data:
    data["conteo_automatico"] = total
    data["densidad"] = (total/((data["longitud"]*1.95))*10000)
    data["conteo_teorico"] = data["longitud"]/0.18
    data["error_pct"] = (np.abs(data["conteo_teorico"]-total)/data["conteo_teorico"])*100
    out.release()

    return data, count_map

def load_workbook(filename):
    df = pd.read_excel(filename)
    df = df.rename(columns={
        "Nombre de video": "video",
        "Fecha": "fecha",
        "Linea": "linea",
        "Longitud (metros)": "longitud",
        "Conteo humano": "conteo_humano",
        "Conteo teorico": "conteo_teorico",
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
        "conteo_teorico": "Conteo teorico",
        "conteo_automatico": "Conteo Automatico",
        "error_pct": "% Error",
        "densidad": "Densidad p/hec"
    })
    df.to_excel(filename, index=False)


def generate_html(data, count_maps):
    max_longitud = max(d["longitud"] for d in data)
    rows = []

    data = sorted(data, key=operator.itemgetter('linea'))

    for i, d in enumerate(data):
        count_map = count_maps[i]

        width_percent = (d["longitud"] / max_longitud) * 100
        densidad = (d["conteo_automatico"] / ((d["longitud"] * 1.95)) * 10000)
        densidad_0 = ((d["longitud"] / 0.18) / ((d["longitud"] * 1.95)) * 10000)

        pct_error = ((densidad - densidad_0) / densidad_0) * 100

        color = "green"
        if -15.0 < pct_error < -10.0:
            color = "yellow"
        elif pct_error <= -15.0:
            color = "red"

        # --- build ticks (evenly spaced) ---
        ticks_html = []
        n = len(count_map)

        for j, entry in enumerate(count_map):
            pos_percent = (j / (n - 1))
            spacing_px = 25
            pos_px = j*spacing_px

            ticks_html.append(f"""
                <div class="tick" style="left: {pos_px}px;">
                    <div class="tick-mark"></div>
                    <div class="tick-label">{entry['id']}</div>
                </div>
            """)

        # --- row ---
        #pxl_width = len(count_map)*30
        bar_pixel_width = max(width_percent, (len(count_map) - 1) * spacing_px)
        row = f"""
        <div class="row">
            <div class="linea">{d['linea']}</div>
            <div class="scroll-wrapper">
                <div class="bar-container">
                    <div class="line" style="width: {bar_pixel_width}px; background-color: {color};">
                        {''.join(ticks_html)}
                    </div>
                </div>
            </div>
            <div class="stats">
                conteo: {d['conteo_automatico']} |
                densidad (contado/deseado): {densidad:.2f}/{densidad_0:.2f}
            </div>
        </div>
        """

        rows.append(row)

    # --- full HTML ---
    html = f"""
    <html>
    <head>
        <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
        }}

        .row {{
            display: flex;
            align-items: center;
            margin: 10px 0;
            width: 100%;
        }}

        .linea {{
            width: 60px;
            text-align: right;
            padding-right: 10px;
            font-weight: bold;
            font-size: 14px;
            flex-shrink: 0;
        }}

        .scroll-wrapper {{
            flex: 1;                 /* fills remaining space */
            overflow-x: auto;
        }}

        .bar-container {{
            position: relative;
            min-width: 100%;         /* at least fill visible area */
            height: 40px;
            background: #eee;
        }}

        .line {{
            position: relative;      /* anchor ticks */
            height: 2px;
            top: 20px;
        }}

        .tick {{
            position: absolute;
            transform: translateX(-50%);  /* center on % */
            top: -10px;
            text-align: center;
        }}

        .tick-mark {{
            width: 1px;
            height: 6px;
            margin: 0 auto;
            background: #2e7d32;
        }}

        .tick-label {{
            font-size: 10px;
            white-space: nowrap;
            color: #2e7d32;
            transform: rotate(-45deg);
            transform-origin: left top;
            margin-top: 2px;
        }}

        .stats {{
            font-size: 14px;
            margin-left: 10px;
            flex-shrink: 0;
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
    count_maps = list()
    for i in range(len(data)):
        # skip those that we've already predicted
        data[i]["conteo_teorico"] = data[i]["longitud"]/0.18
        #if not np.isnan(data[i]["conteo_automatico"]):
            #data[i]["densidad"] = (data[i]["conteo_automatico"]/((data[i]["longitud"]*1.95))*10000)
            #data[i]["conteo_teorico"] = data[i]["longitud"]/0.18
            #data[i]["error_pct"] = (np.abs(data[i]["conteo_teorico"]-data[i]["conteo_automatico"])/data[i]["conteo_teorico"])*100
            #continue 
        model = YOLO(model_path)
        data[i], count_map = track_video(model, base_path, data[i])
        count_maps.append(count_map)
        save_workbook(data, wb_path)
    save_workbook(data, wb_path)
    
    # create the html report
    html = generate_html(data, count_maps)
    with open(os.path.join(base_path, 'data.html'), 'w') as fptr:
        fptr.write(html)

