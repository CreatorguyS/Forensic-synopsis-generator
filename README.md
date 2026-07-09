# 🕵️ Forensic Synopsis Generator & Master Controller

An advanced AI-powered forensic video analysis platform that coordinates multiple state-of-the-art computer vision models running on Kaggle GPUs, processes bounding boxes, computes saliency intervals, and compiles key suspect video tracks.

## 🚀 Key Features
- **Centralized Model Control**: Seamlessly orchestrates and triggers three distinct AI models (YOLOv8 + EffNet + Indexing, DeepFace, and Best Retrieval models) on Kaggle Cloud GPUs.
- **Smart Suspect Zoom & Track Crops**: Extracts suspect tracks and aligns candidate matching crops into clean, uniform galleries using custom blurred-background paddings.
- **Temporal Synopsis Generation**: Automatically detects target visibility intervals and constructs condensed video summaries (synopses).
- **Audio Synchronization**: Re-integrates original scene audio with synopses by dynamically cutting and stitching audio streams matching target interval timestamps using FFmpeg.
- **Premium Frontend UX**: Responsive dark/light UI built with Streamlit, styled with custom CSS card containers, hover micro-animations, and fluid transitions.

---

## 🛠️ Tech Stack & Requirements
* **Frontend UI**: [Streamlit](https://streamlit.io/)
* **Backend Processing**: Python 3.9+, OpenCV, Pandas, SciPy, MoviePy, Kaggle CLI
* **Audio/Video Transcoding**: FFmpeg
* **Cloud GPU Inference**: Kaggle API (Kernels Push)

---

## ⚙️ Setup & Deployment

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/CreatorguyS/Forensic-synopsis-generator.git
   cd Forensic-synopsis-generator
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install system-level dependency `ffmpeg`.
4. Place your Kaggle API token (`kaggle.json`) in `~/.kaggle/`.
5. Run the web application:
   ```bash
   streamlit run app.py
   ```

### ☁️ Streamlit Community Cloud Deployment
1. Log in to [Streamlit Share](https://share.streamlit.io/) using GitHub.
2. Select your repository, set the branch to `main`, and the Entry point to `app.py`.
3. Open **Advanced Settings** -> **Secrets** and configure your Kaggle credentials:
   ```toml
   KAGGLE_USERNAME = "your_kaggle_username"
   KAGGLE_KEY = "your_kaggle_api_key"
   ```
4. Click **Deploy**. Streamlit will automatically read `packages.txt` to install `ffmpeg` and `requirements.txt` for Python packages.
