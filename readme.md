# Multi-Modal Video Summarization via Acoustic-Engagement Fusion

### **KTU Honors Mini-Project | B.Tech CSE**
This project implements an automated video summarization pipeline that identifies high-interest segments in long-form videos by fusing acoustic energy analysis with programmatic video synthesis.

## 🚀 Key Features
* **Acoustic-Driven Detection**: Utilizes Short-Time Energy (STE) to identify "excitement peaks" such as crowd cheers or commentator reactions.
* **Adaptive Thresholding**: Dynamically calculates interest thresholds based on the unique audio profile of each video, ensuring robustness across different recording environments.
* **Multi-Input Support**: Designed to handle local MP4 uploads and processed YouTube streams.
* **Lightweight Architecture**: Optimized to run on standard consumer hardware without the need for high-end GPUs by utilizing Digital Signal Processing (DSP) techniques.

## 🛠️ Tech Stack
* **Backend**: Python, Flask
* **Signal Processing**: Librosa, NumPy
* **Video Engineering**: MoviePy
* **Data Handling**: Pandas
* **Frontend**: HTML5, CSS3, JavaScript

## 📋 Methodology
The system follows a modular four-stage pipeline:
1. **Extraction**: Audio is decoupled from the video source and normalized for analysis.
2. **Signal Analysis**: The audio is processed in temporal windows (5s) to calculate the Short-Time Energy (STE) profile.
3. **Adaptive Filtration**: A dynamic thresholding algorithm identifies segments where energy exceeds the calculated "excitement" baseline.
4. **Programmatic Synthesis**: Overlapping high-energy segments are merged and programmatically subclipped using MoviePy to create a final highlight reel.

## ⚙️ Installation & Setup
1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/nanubee/Multi-Modal-Video-Summarization.git](https://github.com/nanubee/Multi-Modal-Video-Summarization.git)
   cd Multi-Modal-Video-Summarization
2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
3. **Install Dependencies**:
   ```bash
    pip install -r requirements.txt librosa pandas moviepy
4. **Run the Application**:
   ```bash
    python main.py
5. Access the UI at: http://127.0.0.1:5000

### 🎓 Honors Project Context

Developed as part of the **KTU Honors** curriculum at **Rajagiri School of Engineering & Technology (RSET)**, this project explores efficient alternatives to computation-heavy deep learning models for video summarization. 

The system specifically targets **multi-modal video summarization** by combining audio features—such as **Short-Time Energy (STE)** and **MFCCs**—to identify highlight timestamps without requiring high-end hardware.
