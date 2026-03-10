[README.md](https://github.com/user-attachments/files/25857990/README.md)
# Oil Palm FFB Grading System — Documentation

## Overview

This prototype implements automated oil palm Fresh Fruit Bunch (FFB) grading
based on the **MPOB Oil Palm Fruit Grading Manual, 2nd Edition (2003)**.

The system classifies FFB images into 5 categories:
- Ripe Bunch
- Underripe Bunch
- Unripe Bunch
- Rotten Bunch
- Empty Bunch

---

## 1. Project Structure

```
ffb-grading/
├── dataset/
│   ├── train/
│   │   ├── ripe/           ← JPG/PNG images
│   │   ├── underripe/
│   │   ├── unripe/
│   │   ├── rotten/
│   │   └── empty/
│   ├── val/
│   │   └── (same structure)
│   └── test/
│       └── (same structure)
│
├── output/
│   ├── ffb_model_best.pth  ← saved after training
│   ├── training_history.json
│   └── classification_report.txt
│
├── train_ffb_classifier.py ← training script
├── app.py                  ← Streamlit web interface
├── requirements.txt
└── README.md (this file)
```

---

## 2. Installation

### Requirements
- Python 3.9+
- CUDA-capable GPU recommended (NVIDIA, 4GB+ VRAM) or CPU (slower)

### Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install streamlit pillow scikit-learn numpy
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

### requirements.txt
```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
pillow>=9.0.0
scikit-learn>=1.2.0
numpy>=1.23.0
```

---

## 3. Dataset Preparation

### 3.1 Minimum Dataset Size

| Class       | Minimum Images | Recommended |
|-------------|---------------|-------------|
| Ripe        | 200           | 500+        |
| Underripe   | 200           | 500+        |
| Unripe      | 100           | 300+        |
| Rotten      | 100           | 300+        |
| Empty       | 100           | 300+        |

**Tip:** The model uses data augmentation, so even 100 images per class
can produce a working prototype. More is always better.

### 3.2 Image Collection Guidelines

**Shooting conditions:**
- Photograph bunches on the loading ramp or weighbridge platform
- Natural daylight or consistent artificial lighting
- Avoid heavy shadows across the bunch
- Distance: 0.5–1.5m from bunch (bunch should fill 60%+ of frame)
- Resolution: minimum 640×480px; 1080p preferred
- Format: JPG or PNG

**Angles to capture per bunch:**
- Top view (looking down at the bunch)
- Side view (to see stalk length and fruitlet detachment)
- Close-up of fruitlet colour (critical for ripe vs. underripe)

**What to photograph:**
- The whole bunch clearly visible
- Include some loose fruitlets in the frame if present
- For rotten/empty bunches, ensure the distinguishing features are visible

### 3.3 Classification Reference (MPOB Manual)

When labeling images, apply these MPOB criteria:

| Class      | Key Visual Indicators                                   |
|------------|---------------------------------------------------------|
| Ripe       | Reddish orange; ≥10 fresh sockets; >50% fruits on bunch |
| Underripe  | Reddish orange or purplish red; <10 fresh sockets       |
| Unripe     | Black or purplish black; no fresh sockets               |
| Rotten     | Blackish, mouldy, decomposed                            |
| Empty      | >90% fruitlets detached; bare spikelet structure        |

### 3.4 Free/Low-Cost Image Sources

1. **Your own mill/estate** — most relevant and representative
2. **MPOB publications** — reference photographs
3. **Google Images** — search "ripe oil palm bunch", "empty FFB", etc.
   (use for augmentation/supplementing only; verify each image)
4. **Kaggle** — search "palm oil fruit dataset" for existing labeled datasets
5. **Roboflow Universe** — some agricultural datasets available

### 3.5 Labeling Tools (Free)

- **LabelImg** (`pip install labelImg`) — simple image classification folder organizer
- **Label Studio** — more advanced, web-based, free tier available
- **CVAT** (online) — free open-source annotation tool

For simple classification (no bounding boxes needed), just organize
images into the correct class subfolders.

### 3.6 Train/Val/Test Split

Recommended split: **70% / 20% / 10%**

Quick script to split your dataset:
```python
import os, shutil, random
from pathlib import Path

def split_dataset(source_dir, output_dir, ratios=(0.7, 0.2, 0.1)):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(f"{source_dir}/{d}")]
    splits = ["train", "val", "test"]
    
    for cls in classes:
        imgs = list(Path(f"{source_dir}/{cls}").glob("*.[jJ][pP][gG]"))
        imgs += list(Path(f"{source_dir}/{cls}").glob("*.png"))
        random.shuffle(imgs)
        
        n = len(imgs)
        ends = [int(n * ratios[0]), int(n * (ratios[0]+ratios[1])), n]
        starts = [0] + ends[:-1]
        
        for split, s, e in zip(splits, starts, ends):
            out_dir = Path(output_dir) / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs[s:e]:
                shutil.copy(img, out_dir / img.name)
        
        print(f"{cls}: {n} total → train={ends[0]}, val={ends[1]-ends[0]}, test={n-ends[1]}")

split_dataset("./raw_images", "./dataset")
```

---

## 4. Training the Model

```bash
python train_ffb_classifier.py
```

**What happens:**
1. Epochs 1–5: Only the classification head is trained (backbone frozen)
2. Epoch 6+: Full model fine-tuned at a lower learning rate
3. Best model saved automatically to `./output/ffb_model_best.pth`
4. Early stopping if validation accuracy plateaus

**Expected training time:**
- GPU (RTX 3060 / T4): ~15–30 minutes for 40 epochs
- CPU only: ~2–4 hours

**Expected accuracy (with 200+ images/class):**
- Ripe vs. others: ~90–95%
- Full 5-class: ~75–88% (depends heavily on dataset quality)

---

## 5. Running the Web App

```bash
streamlit run app.py
```

Open browser at: `http://localhost:8501`

**Features:**
- Upload FFB image or use webcam
- Instant classification with confidence scores
- MPOB-based penalty and extraction rate calculation
- Downloadable grading report (.txt)

---

## 6. Improving Accuracy

### If accuracy is low (<75%):
1. **More data** — double the dataset size; this has the biggest impact
2. **Better images** — consistent lighting, no motion blur
3. **Try ResNet50** — change `backbone` in CONFIG to `"resnet50"`
4. **Increase epochs** — set `num_epochs: 60`
5. **Check label quality** — mislabeled images are a major cause of low accuracy

### For production deployment:
1. Add more bunch categories (overripe, long stalk, dirty, dura)
2. Integrate full MPOB penalty tables (Tables III–XI)
3. Add bunch weight estimation using depth camera or scale integration
4. Deploy on edge device (Raspberry Pi 5 + camera) for ramp-side use
5. Add multi-bunch detection using YOLO (detect multiple bunches per image)

---

## 7. Limitations

1. **Single-bunch classification** — currently grades one bunch per image
2. **No weight estimation** — average bunch weight entered manually
3. **5 classes only** — MPOB defines 16 bunch types; others not yet implemented
4. **Prototype accuracy** — performance depends entirely on dataset quality
5. **Requires licensed officer** — per MPOB Section 3.2, AI results must be
   verified by a qualified human grader

---

## 8. Estimated Development Cost & Timeline

### Phase 1: Prototype (this codebase) — 4–6 weeks
- Dataset collection and labeling: 1–2 weeks
- Model training and tuning: 1 week
- Web app development: 1 week
- Testing and validation: 1–2 weeks
- **Estimated cost: RM 8,000–15,000**

### Phase 2: Production System — 3–4 months additional
- Mobile app (React Native or Flutter)
- Full MPOB penalty table integration
- Database for grading records
- YOLO-based multi-bunch detection
- Edge deployment (Jetson Nano or Raspberry Pi)
- Integration with weighbridge system
- **Estimated cost: RM 25,000–50,000**

---

## 9. References

- MPOB Oil Palm Fruit Grading Manual, 2nd Edition (2003)
- ISBN: 967-961-091-8
- Published by Malaysian Palm Oil Board (MPOB)

---

## License

Prototype code — for research and demonstration purposes only.
Not for commercial deployment without modification and validation.
