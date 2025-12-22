# Smart Fish Counting and Sorting System with Yolov11

This project is an advanced computer vision system that uses artificial intelligence to detect fish passing on a production line or conveyor belt, counts them in batches, and assigns them to the appropriate crates based on their exit positions.

The system uses a **Histogram-Based Verification (Batch Correction)** algorithm to prevent instantaneous detection errors (flickering).

## Features

* **Dual Model Architecture:** Two separate YOLOv8 models (`ultralytics`) customized for crates and fish operate simultaneously.
* **Smart Batch Analysis:** Fish are analyzed in batches rather than individually. A statistical histogram is used to eliminate false positives.
* **Drag-and-Drop ROI (Region of Interest):** When the application starts, the user can dynamically define the area to be counted by dragging it with the mouse.
* **Dynamic Thresholding:** Automatically adjusts verification sensitivity (Ratio: 10% or 15%) based on the speed at which fish pass across the screen.
* **Bin Matching:** When the counted fish group leaves the screen, the bin closest to the coordinates where the group was last seen is automatically matched.
* **Detailed Reporting:** At the end of the process, a video recording, bin-based counting report, and algorithmic decision logs are generated.

## 🛠️ Setup and Codes

* Python 3.10 and the following libraries must be installed for the project to work.
* The utils file contains utility code that I use when preparing data for the model.

### Requirements (For Python3.10)
```bash
pip install -r requirements.txt
```

## Demo
***Here is an example of how the code works.***
![Demo](demo.gif)

# Note
**Since this project is an academic study, I couldn't upload the .pt files I trained and used. But you can review the 'detect' in the Result folder.**