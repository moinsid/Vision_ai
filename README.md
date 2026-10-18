# Vision AI – Real-Time Object Detection

A real-time object detection project built using Python, OpenCV, and the Ultralytics YOLO26 model. The application captures live video from a webcam, detects objects in each frame, and displays the results with bounding boxes and labels.

## 🚀 Features

- Real-time object detection using a webcam.
- Object recognition powered by YOLO26.
- Live bounding boxes and object labels.
- Video capture and display using OpenCV.
- Press `q` to exit the application.

## 🛠️ Technologies Used

- **Python** – Programming language.
- **Ultralytics YOLO26** – Object detection model.
- **OpenCV** – Webcam access and video processing.

## 📂 Project Structure

```text
vision_ai/
├── main.py
├── yolo26n.pt
├── images.jpg
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/vision_ai.git
cd vision_ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

For Windows:

```bash
.venv\Scripts\activate
```

### 4. Install the required libraries

```bash
pip install ultralytics opencv-python
```

## ▶️ How to Run

Run the following command:

```bash
python main.py
```

Make sure your webcam is connected and accessible. The application will open a window showing the live camera feed with detected objects.

Press **`q`** to stop the program and close the window.

## 🧠 How It Works

1. Loads the pretrained YOLO26 model from `yolo26n.pt`.
2. Opens the default webcam using `cv2.VideoCapture(0)`.
3. Reads video frames continuously.
4. Passes each frame to the YOLO model for object detection.
5. Draws bounding boxes and labels using `results[0].plot()`.
6. Displays the processed frames using OpenCV.

## 🔮 Future Enhancements

- Add object tracking.
- Display detection confidence scores.
- Support image and video file detection.
- Add a graphical user interface.
- Improve detection performance.

## 👨‍💻 Author

**Moinuddin Siddiqui**

GitHub: [@moinsid](https://github.com/moinsid)

---

If you find this project useful, feel free to ⭐ the repository!


If you find this project useful, consider giving the repository a ⭐ on GitHub!
