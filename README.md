# 🚦 Object Detection Using MobileNet SSD with OpenCV

This project demonstrates **Object Detection** on static images using **MobileNet SSD (Single Shot Detector)** architecture with **OpenCV's DNN module** and **TensorFlow-trained models**.  
It uses the **COCO dataset** classes for object detection like person, car, bicycle, etc.

---

## 📂 Project Structure
```
.
├── models/
│   └── ssd_mobilenet_v2_coco_2018_03_29/
│       ├── frozen_inference_graph.pb
│       └── ssd_mobilenet_v2_coco_2018_03_29.pbtxt
├── coco_class_labels.txt
├── opencv_bootcamp_assets_NB13.zip  # Dataset (downloaded via code)
├── street1.jpg                      # Input Image
├── object_detection.py              # Main Code
└── README.md
```

---
Models must be downloaded manually. Only the folder names and file placeholders are included.
✅ frozen_inference_graph.pb
✅ ssd_mobilenet_v2_coco_2018_03_29.pbtxt
---

## 🚀 How It Works

### 🔹 1. **Model**
We use **MobileNet SSD V2** pre-trained on **MS-COCO dataset**.  
`ssd_mobilenet_v2_coco_2018_03_29.pb` is the TensorFlow frozen model.  
`ssd_mobilenet_v2_coco_2018_03_29.pbtxt` is the configuration for OpenCV.

### 🔹 2. **Pipeline Steps**
| Step             | Description                                  |
|------------------|----------------------------------------------|
| Read Image       | Load an image (street1.jpg)                  |
| Blob Creation    | `cv2.dnn.blobFromImage` to preprocess image  |
| Load Model       | Load TensorFlow model via OpenCV DNN          |
| Inference        | Forward pass: detect objects                 |
| Post-process     | Extract boxes, class labels, confidence       |
| Display Results  | Draw boxes and labels using OpenCV / matplotlib |

---

## 🔧 Requirements
- **Python 3.x**
- **OpenCV (4.x recommended)**  
- **NumPy**
- **Matplotlib**

Install dependencies:
```bash
pip install opencv-python numpy matplotlib
```

---

## 📥 Download Assets
The dataset and models are automatically downloaded via:
```python
download_and_unzip(URL, asset_zip_path)
```
No manual download is required.

---

## 📄 Files Explained

| File                 | Purpose                                      |
|-----------------------|----------------------------------------------|
| `object_detection.py` | Main Python code for object detection        |
| `coco_class_labels.txt` | Contains 80 COCO class labels (person, car, dog, etc.) |
| `models/`             | Pre-trained MobileNet SSD model (frozen graph) |
| `street1.jpg`         | Sample test image                            |

---

## 🔍 Key Functions

### 1️⃣ `detect_objects(net, im, dim=300)`
- Converts image to blob.
- Feeds blob to network.
- Returns detected objects.

### 2️⃣ `display_objects(im, objects, threshold=0.25)`
- Extracts class ID, confidence, bounding boxes.
- Displays objects with labels if confidence > threshold.

### 3️⃣ `display_text(im, text, x, y)`
- Utility to draw background box and label on image.

---

## 🎨 Sample Output
| Input Image     | Output with Detected Objects   |
|-----------------|--------------------------------|
| ![Input](assets/input.jpg) | ![Output](assets/output.jpg) |

*(Replace these with your actual output images in GitHub repo)*

---

## 📚 Concepts Covered
✅ Object Detection  
✅ DNN with OpenCV  
✅ TensorFlow Pre-trained Models  
✅ Image Preprocessing (Blob)  
✅ COCO Dataset Labels  

---

## 📌 Why MobileNet-SSD?
- **Lightweight:** Suitable for real-time applications.
- **Pre-trained:** COCO dataset for common objects.
- **Fast:** Optimized for edge devices.

---

## 🛠️ Possible Extensions
- Real-time webcam object detection.
- Video input support.
- Different models like YOLO or Faster R-CNN.
- Performance benchmarking.

---

## 📸 Example Classes Detected
✔ Person  
✔ Bicycle  
✔ Car  
✔ Bus  
✔ Motorcycle  
✔ Dog  
✔ Chair  
✔ etc. (80 COCO classes)

---

## ✍️ Author
**Nandini Sharma**  
OpenCV & Computer Vision Enthusiast  
[LinkedIn Profile](https://www.linkedin.com/in/YOUR-LINKEDIN-HERE) *(Update this link)*

---

## ⭐ License
This project is for **educational purposes** only.







