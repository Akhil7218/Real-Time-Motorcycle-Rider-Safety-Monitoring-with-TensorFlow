# Real-Time-Motorcycle-Rider-Safety-Monitoring-with-TensorFlow
Real-time motorcyclist helmet detection using TensorFlow and YOLO, saving images of riders without helmets for improved road safety.


## Table of Contents
* [About the Project](#about-the-project)
* [Prerequisites](#prerequisites)
* [Dataset](#dataset)
* [High-Level System Architecture](#high-level-system-architecture)
* [Training](#training)
* [Implementation](#implementation)
* [Results](#results)
* [Contributing](#contributing)

***

## About the Project
This project uses a two-stage approach to detect motorcyclists without helmets in both images and videos. The first stage identifies motorcyclists in a given frame using a pre-trained **TensorFlow Object Detection API** model. The second stage then crops the detected motorcyclist and uses a custom-trained **YOLOv3** model to determine whether they're wearing a helmet or not. The final output provides a bounding box around the motorcyclist, colored red for "No Helmet" and green for "Helmet".

## Prerequisites
The following open-source packages are required to run this project:
* Python
* OpenCV
* Numpy
* Matplotlib
* TensorFlow
* TensorFlow Object Detection API

***

## Dataset
This project uses two separate datasets for training the two different detection models. Note that only a subset of the files is included in the project repo due to size restrictions.

### Bike Rider Dataset
A custom dataset of **739 images** was created by scraping images from Google. These images, which include riders with and without helmets, were then annotated using the LabelImg tool.

### Helmet Dataset
A dataset of **764 images** containing two distinct classes ("Helmet" and "No Helmet") was taken from [Kaggle](https://www.kaggle.com/). The bounding box annotations for this dataset are provided in the **XML format**.

***

## High-Level System Architecture
The system follows a two-stage pipeline:

1.  **Motorcyclist Detection:** The first stage uses a **TensorFlow Object Detection API** model, specifically a pre-trained `frozen_inference_graph.pb` model, to detect motorcyclists. It identifies the bounding boxes for all motorcyclists in an image or video frame.
2.  **Helmet Detection:** The cropped images of the detected motorcyclists from the first stage are then passed to the second stage. This stage uses a custom-trained **YOLOv3** model (`yolov3_custom_4000.weights`) to classify the cropped region as either "Helmet" or "No Helmet".

***

## Training
The project uses pre-trained models, but the second stage utilizes a custom-trained **YOLOv3** model. The `.cfg` and `.weights` files (`yolov3_custom.cfg` and `yolov3_custom_4000.weights`) are used for this.

## Implementation
The implementation is structured in a Jupyter Notebook (`detection.ipynb`) and consists of the following key steps:

1.  **Setup and Cloning:** The project repository and the TensorFlow models are cloned. The necessary libraries like `tf_slim` and `protobuf` are installed.
2.  **Model Loading:** The pre-trained TensorFlow detection graph and the custom YOLOv3 model are loaded from their respective paths (`rcnn/frozen_inference_graph.pb` and `yolo/yolov3_custom_4000.weights`).
3.  **Inference:**
    * **Image Processing:** For images, the input is resized and expanded to match the model's input dimensions.
    * **Frame Processing:** For videos, each frame is read, resized, and processed. Frames are skipped (e.g., every third frame) to improve performance.
4.  **Detection and Classification:**
    * The TensorFlow model performs initial detection, providing bounding boxes for motorcyclists.
    * These bounding boxes are then used to crop the image, and the cropped sections are fed into the YOLOv3 model for helmet detection.
5.  **Output Generation:** Bounding boxes are drawn on the original image or video frame, colored green for "Helmet" and red for "No Helmet". The processed images and videos are saved to the output folder.

***

## Results
The system successfully identifies motorcyclists and classifies whether they are wearing a helmet or not. Images of motorcyclists without helmets are saved to the output folder. For videos, a new video file with the detections is created.

***

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature/your-feature`).
3.  Make your changes and commit them (`git commit -m 'Add your feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Create a new Pull Request.
