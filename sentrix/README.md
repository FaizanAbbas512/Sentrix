# SENTRIX — Edge Surveillance Intelligence

Lightweight, real-time **loitering**, **abandoned object**, aur **crowd density**
detection — Pakistani Safe-City jaise resource-constrained surveillance ke liye.
YOLO26 (2026) detection + ByteTrack tracking + custom behavioral analyzers +
ek live web dashboard.

> Yeh project ek research paper ke liye banaya gaya hai. Code modular hai,
> har module ka apna kaam hai, aur `evaluate.py` se paper ke numbers nikalte hain.

---

## 1. Folder Structure

```
sentrix/
├── app.py                  # Web dashboard (Flask) — main entry
├── run_cli.py              # Bina browser, seedha OpenCV window
├── evaluate.py             # Paper ke metrics (FPS, latency, counts)
├── config.yaml             # Saari settings — sirf yahan badlo
├── requirements.txt
│
├── src/
│   ├── detector.py         # YOLO26 + tracking wrapper
│   ├── pipeline.py         # Sab kuch jodne wali pipeline + drawing
│   ├── video.py            # Webcam / video file handler
│   ├── alerts.py           # Cooldown + CSV log + snapshots
│   └── analyzers/
│       ├── loitering.py    # Ek jagah zyada der = alert
│       ├── abandoned.py    # Lawaaris bag = alert
│       └── crowd.py        # Bheed = alert
│
├── dashboard/
│   ├── templates/index.html
│   └── static/css/style.css, js/app.js
│
├── data/
│   ├── videos/             # Yahan test videos rakho
│   └── logs/               # alerts.csv + snapshots/ (auto-generated)
│
└── results/                # Evaluation reports (paper ke liye)
```

---

## 2. Setup

Tera `cctv_ai` venv already bana hua hai. Usi mein:

```bash
cctv_ai\Scripts\activate
pip install -r requirements.txt
```

(Flask aur pyyaml naye honge, baaki already installed hain.)

---

## 3. Chalana

**A) Web Dashboard (recommended — demo video ke liye best):**
```bash
python app.py
```
Phir browser kholo: `http://127.0.0.1:5000`

**B) Quick test (bina browser):**
```bash
python run_cli.py
```
Window pe `q` dabao band karne ke liye.

**C) Paper ke numbers nikaalo:**
```bash
python evaluate.py --video data/videos/test.mp4
```
`results/eval_report.txt` mein FPS, latency, counts aa jayenge.

---

## 4. Settings (config.yaml)

Sab kuch `config.yaml` se control hota hai — code touch nahi karna:

| Setting | Matlab |
|---|---|
| `source` | `0` = webcam, ya video file ka path |
| `loitering.dwell_seconds` | Itni der ek jagah = loitering |
| `abandoned.unattended_seconds` | Object akela itni der = abandoned |
| `abandoned.owner_radius` | Is radius mein person ho to "attended" |
| `crowd.threshold` | Itne se zyada log = crowd alert |

Test ke liye webcam pe `dwell_seconds` ko `5` rakho taake jaldi alert dikhe.

---

## 5. Research / Paper Guidance

### Tera contribution (paper mein yeh likhna)
Existing kaam (jaise Natha et al., 2024 — YOLOv8+Transformer, *road* anomalies:
accidents, snatching) ne **dynamic road events** pe focus kiya. SENTRIX
**static/temporal anomalies** (loitering, abandoned objects) pe focus karta hai,
**latest YOLO26** use karta hai, aur **edge/CPU pe real-time** chalta hai —
jo Pakistan ki low-power, load-shedding waali surveillance reality ke liye
zaroori hai.

### Paper ka structure (6–8 pages)
1. **Abstract** — kya banaya, kya results.
2. **Introduction** — Pakistani Safe City context, power/bandwidth constraints,
   kyon lightweight edge detection chahiye.
3. **Related Work** — Natha et al. (YOLOv8+Transformer), classic VAD, ViT-based
   methods. Saaf likhna ke woh road events pe the, tu static anomalies pe hai.
4. **Methodology** — yeh architecture (detector → tracker → 3 analyzers →
   alerts). Diagrams `pipeline.py` ke flow se banao.
5. **Experiments** — `evaluate.py` ke results: FPS, latency, detection counts.
   Compare karo YOLO26-n vs YOLO26-s (config mein weights badal ke dono chalao).
6. **Results & Discussion** — real-time chal raha (FPS > 15?), false positive
   discussion, snapshots dikhao (`data/logs/snapshots/`).
7. **Limitations & Future Work** — crowd density map, multi-camera, occlusion.
8. **Conclusion**.

### Experiments jo abhi chala sakte ho (free, laptop pe)
- **Model size comparison:** `config.yaml` mein `weights: yolo26n.pt` phir
  `yolo26s.pt` — dono pe `evaluate.py` chalao, FPS vs accuracy table banao.
- **Frame width vs speed:** `frame_width` ko 640 / 960 / 1280 — FPS measure karo.
- **Threshold sensitivity:** `dwell_seconds` badal ke false-positive rate dekho.
- **Test footage:** YouTube se Pakistani street/CCTV clips download karke
  `data/videos/` mein daalo (research/educational use).

### Datasets (real footage ke liye)
- **RAD (Road Anomaly Dataset)** — Mendeley/Kaggle pe free, Pakistani context.
- **UCF-Crime** — international benchmark, comparison ke liye.
- Apni webcam se khud ke loitering/abandoned clips record kar sakte ho.

### Publish
- Paper Overleaf pe likho (free), phir **arXiv** pe submit (free).
- Code **GitHub** pe public karo — paper mein link do. Yeh Erasmus/MEXT/CSC
  applications ko strong karta hai.

---

## 6. Notes
- Pehli baar chalane pe YOLO26 weights auto-download honge (~6 MB for nano).
- GPU optional hai — CPU pe bhi chalega (nano model isi liye choose kiya).
- Snapshots aur CSV log automatically `data/logs/` mein save hote hain.
