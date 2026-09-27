# ♻️ Intelligent Waste Classification and Segmentation

An end-to-end **Deep Learning and Computer Vision application** for identifying, segmenting, classifying, and counting common waste materials from images.
The project uses two specialized YOLO models in a single Streamlit application:

- **Waste Pile Mode:** Uses YOLO11 instance segmentation to detect and segment multiple waste objects in a mixed pile and count each category.
- **Single Item Mode:** Uses a dedicated YOLO11 classification model to classify an isolated waste item such as plastic, paper/cardboard, metal, glass, or household waste.

## 🚀 Project Overview

Manual waste sorting can be time-consuming, especially when different types of waste are mixed together. This project demonstrates how deep learning and computer vision can be used to automate part of the waste-analysis process.
The system supports two different scenarios:

### 1. Single Waste Item

Upload an image containing one waste item.

```
Image
  ↓
YOLO11 Classification Model
  ↓
Waste Category + Confidence
```

Example:

```
Input: Glass bottle

Prediction:
Glass
Confidence: 97.15%
```

### 2. Mixed Waste Pile

Upload an image containing multiple waste objects.

```
Waste Pile Image
       ↓
YOLO11 Segmentation Model
       ↓
Object Detection + Segmentation
       ↓
Category Classification
       ↓
Waste Category Counts
```

The application displays the original image, annotated segmentation output, and the number of detected objects in each category.

---

## 🧠 Models

The project uses two separate models because single-item classification and mixed-pile segmentation are different computer vision tasks.

| Model       | Task                  | Purpose                       |
| ----------- | --------------------- | ----------------------------- |
| YOLO11s-seg | Instance Segmentation | Initial segmentation baseline |
| YOLO11m-seg | Instance Segmentation | Final mixed waste-pile model  |
| YOLO11s-cls | Image Classification  | Final single-item classifier  |

### Waste Categories

The final system supports five categories:

- Glass
- Household Waste
- Metal
- Paper/Cardboard
- Plastic

The original dataset also contained a **biowaste** category, but it had no training annotations and was therefore excluded from the final five-class models.

---

## 📊 Dataset

The project uses the **Kaggle Garbage Segmentation / Dumpster Diving dataset**.
The training data contains:

- **941 training images**
- **2,461 annotated waste objects**
- COCO-style polygon segmentation annotations

### Object Distribution

| Category        |   Objects |
| --------------- | --------: |
| Glass           |       291 |
| Household Waste |       668 |
| Metal           |       417 |
| Paper/Cardboard |       361 |
| Plastic         |       724 |
| **Total**       | **2,461** |

For the segmentation workflow, the data was split at image level into:

- **752 training images**
- **189 validation images**

---

## 🔧 Data Preprocessing

### Segmentation Dataset

The original COCO-style polygon annotations were converted into YOLO segmentation format.
The class mapping is:

```
0 → glass
1 → household_waste
2 → metal
3 → paper_cardboard
4 → plastic
```

### Single-Item Classification Dataset

The segmentation polygons were also used to create a separate classification dataset.
For each annotated object:

1. The segmentation polygon was converted into a mask.
2. The object was isolated from the original image.
3. The object was cropped using its bounding region.
4. Pixels outside the object were replaced with a white background.
5. A small padding margin was added.
6. The resulting image was saved into its corresponding class folder.

This produced:

- **1,966 training object images**
- **495 validation object images**
- **2,461 total single-item samples**

---

## 📈 Model Results

### YOLO11s Segmentation

The initial YOLO11 small segmentation model achieved approximately:

```
Mask mAP50:      0.304
Mask mAP50-95:   0.221
```

### YOLO11m Segmentation

The final YOLO11 medium segmentation model achieved approximately:

```
Mask mAP50:      0.317
Mask mAP50-95:   0.224
```

Approximate per-class mask mAP50:

| Category        | Mask mAP50 |
| --------------- | ---------: |
| Glass           |      0.339 |
| Household Waste |      0.090 |
| Metal           |      0.383 |
| Paper/Cardboard |      0.266 |
| Plastic         |      0.502 |

### Single-Item Classifier

The YOLO11 small classification model achieved:

```
Top-1 Accuracy: 56.6%
Top-5 Accuracy: 100%
```

Because the classification problem contains exactly five classes, Top-5 accuracy is not a useful practical metric. Top-1 accuracy is the important metric for this application.
A real-world glass-bottle test image was correctly classified as:

```
Glass
Confidence: 97.15%
```

---

## 🧪 Testing

The models were tested using unseen images.
The segmentation model was able to detect multiple objects in mixed waste-pile images and produce:

- Bounding boxes
- Segmentation masks
- Class labels
- Confidence scores
- Category counts

The single-item classifier was tested on individual waste images.

### Example Failure Case

During testing, a crumpled paper image was incorrectly classified as plastic by the original segmentation model.
This showed an important limitation of using the pile-segmentation model for isolated-item classification.
To address this, a separate single-item classification model was developed.

---

## 🖥️ Streamlit Application

The final application provides two modes:

### Single Item

```
Upload single item
       ↓
Classification model
       ↓
Category + Confidence
```

### Waste Pile

```
Upload waste pile
       ↓
Segmentation model
       ↓
Annotated image
       ↓
Category counts
```

---

## 📁 Project Structure

```
waste_streamlit_app/
│
├── app.py
├── best.pt
├── single_item_classifier.pt
├── requirements.txt
└── README.md
```

### Files

| File                        | Description                              |
| --------------------------- | ---------------------------------------- |
| `app.py`                    | Streamlit application                    |
| `best.pt`                   | YOLO11m waste-pile segmentation model    |
| `single_item_classifier.pt` | YOLO11s single-item classification model |
| `requirements.txt`          | Python dependencies                      |
| `README.md`                 | Project documentation                    |

> The original training dataset is not required to run the Streamlit inference application.

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/Edwinkjohnson/Waste_Analysis_Computer_Vision/
```

### 2. Open the project directory

```
cd waste_streamlit_app
```

### 3. Create a virtual environment

```
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```
venv\Scripts\activate
```

#### Linux / macOS

```
source venv/bin/activate
```

### 5. Install dependencies

```
python -m pip install -r requirements.txt
```

### 6. Run the Streamlit application

```
python -m streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

The main technologies used in this project are:

- Python
- YOLO11
- Ultralytics
- PyTorch
- OpenCV
- NumPy
- Pillow
- Pandas
- Streamlit

---

## 🛠️ Development Workflow

The project followed this workflow:

```
Kaggle Dataset
      ↓
Exploratory Data Analysis
      ↓
COCO Annotation Analysis
      ↓
Segmentation Dataset Preparation
      ↓
YOLO11 Segmentation Training
      ↓
Segmentation Evaluation
      ↓
Single-Object Extraction
      ↓
Classification Dataset Creation
      ↓
YOLO11 Classification Training
      ↓
Classification Evaluation
      ↓
Unseen Image Testing
      ↓
Streamlit Application
```

---

## ⚠️ Limitations

This project is a research/academic prototype and should not be considered a fully reliable automated waste-sorting system.
Current limitations include:

- Single-item Top-1 validation accuracy is 56.6%.
- Segmentation performance is moderate.
- Household waste is particularly difficult for the segmentation model.
- Small and overlapping objects can be difficult to segment.
- The dataset does not cover every possible real-world waste item.
- The model supports only the five trained categories.
- Classification performance may vary with lighting, background, object orientation, and object appearance.

---

## 🔮 Future Improvements

Possible future improvements include:

- Analyze the classification confusion matrix.
- Clean and review incorrect or ambiguous object crops.
- Collect more real-world single-item images.
- Improve class balance.
- Use stronger data augmentation.
- Train a larger classification model such as YOLO11m-cls.
- Train with GPU resources.
- Experiment with higher image resolutions.
- Improve segmentation on crowded and overlapping scenes.
- Add CSV report generation.
- Add downloadable annotated images.
- Deploy the application online.

---

## 🎯 Project Objective Alignment

This project addresses the main requirements of the Computer Vision project brief:

| Requirement                       | Completed |
| --------------------------------- | --------- |
| Real-world problem identification | ✅         |
| Dataset sourcing                  | ✅         |
| Exploratory Data Analysis         | ✅         |
| Data preprocessing                | ✅         |
| Deep learning model selection     | ✅         |
| Model training and fine-tuning    | ✅         |
| Model evaluation                  | ✅         |
| Testing on unseen images          | ✅         |
| User interface development        | ✅         |
| Streamlit deployment prototype    | ✅         |
| Project report                    | ✅         |

---

## 📚 References

- Kaggle – Garbage Segmentation / Dumpster Diving Dataset
  [https://www.kaggle.com/competitions/garbage-segmentation/data](https://www.kaggle.com/competitions/garbage-segmentation/data)

- Ultralytics YOLO Documentation
  [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

- Streamlit Documentation
  [https://docs.streamlit.io/](https://docs.streamlit.io/)

- Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. ECCV, 2014.

---

## 👨‍💻 Project

**Intelligent Waste Classification and Segmentation Using Deep Learning**
Developed as a Deep Learning and Computer Vision project demonstrating dataset preparation, model training, evaluation, image classification, instance segmentation, and deployment with Streamlit.
