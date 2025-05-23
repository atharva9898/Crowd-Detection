# Crowd Detection Using Computer Vision 🧠🎥

This project detects people in a video and identifies when **3 or more individuals are standing close together for 10 consecutive frames** using **YOLOv5** and **OpenCV**.

## 📽️ Demo
Check out the [demo video](assets/dataset_video.mp4) showing crowd detection in real-time.

## 🚀 Features
- Person detection using YOLOv5
- Distance-based proximity analysis
- Frame-by-frame group tracking
- Logging events where crowds form
- Annotated video output

## 📁 Folder Structure
crowd-detection/

├── crowd_detection.py # Main detection script

├── assets/demo_video.mp4 # Original input video

├── outputs/annotated_video.mp4 # Output with bounding boxes and logs

├── outputs/crowd_log.csv # Log of crowd events

├── requirements.txt # Python dependencies


## 🛠️ Tech Stack
- Python
- OpenCV
- YOLOv5
- NumPy

## 🧪 How to Run
1. Clone the repository:
```bash
git clone https://github.com/your-username/crowd-detection.git
cd crowd-detection
```
2. Install Dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Detection:
```bash
python crowd_detection.py --video assets/demo_video.mp4
```

Output will be saved to outputs/.

📊 Output

annotated_video.mp4 — Visual bounding boxes and group tags

crowd_log.csv — Timestamped records of crowd formations

📌 Use Cases

Public safety & surveillance

Event management

Social distancing enforcement

📩 Contact
For questions or collaborations, feel free to reach out:

Atharva Chavan

LinkedIn - https://www.linkedin.com/in/atharva-chavan-ab891a203/

Email - atharva.chavan9898@gmail.com

