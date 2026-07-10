import streamlit as st
import subprocess
import time
import os
import json
import threading
import queue
import sys
from uuid import uuid4

os.environ["PYTHONIOENCODING"] = "utf-8"

# --- Kaggle API Credentials Setup for Cloud Deployment ---
if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
elif "kaggle" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"].get("KAGGLE_USERNAME", "")
    os.environ["KAGGLE_KEY"] = st.secrets["kaggle"].get("KAGGLE_KEY", "")


def resize_and_pad(img, target_width=200, target_height=250):
    import cv2
    import numpy as np
    
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
        
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create the background canvas by resizing the original crop to target size and blurring it heavily
    bg = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (21, 21), 0)
    # Dim the background to make the foreground pop
    bg = cv2.convertScaleAbs(bg, alpha=0.4, beta=0)
    
    # Paste the resized image in the center
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return bg

def extract_top_frames(job_dir, vid_name, target_id, top_k=10):
    original_video = os.path.join(job_dir, "inputs", vid_name)
    output_dir = os.path.join(job_dir, "outputs")
    
    # Locate tracked_persons.csv
    csv_path = None
    for root, _, files in os.walk(output_dir):
        if "tracked_persons.csv" in files:
            csv_path = os.path.join(root, "tracked_persons.csv")
            break
            
    if not csv_path or not os.path.exists(csv_path) or not os.path.exists(original_video):
        return []
        
    try:
        import pandas as pd
        import cv2
        
        df = pd.read_csv(csv_path)
        # Filter for the target suspect
        df_target = df[df["track_id"] == target_id].copy()
        if df_target.empty:
            return []
            
        # Sort by yolo_conf descending to get highest confidence frames
        df_target = df_target.sort_values(by="yolo_conf", ascending=False)
        
        # Filter to keep frames spaced apart (at least 15 frames / 0.5 seconds apart) to avoid redundancy
        selected_rows = []
        min_frame_distance = 15
        for idx, row in df_target.iterrows():
            f_idx = int(row["frame_index"])
            too_close = False
            for sel_row in selected_rows:
                if abs(f_idx - int(sel_row["frame_index"])) < min_frame_distance:
                    too_close = True
                    break
            if not too_close:
                selected_rows.append(row)
                if len(selected_rows) >= top_k:
                    break
        
        top_frames = []
        cap = cv2.VideoCapture(original_video)
        
        # Directory to save extracted crops
        crops_dir = os.path.join(output_dir, "top_crops")
        # Clear existing crops to prevent showing outdated/redundant ones
        if os.path.exists(crops_dir):
            import shutil
            shutil.rmtree(crops_dir)
        os.makedirs(crops_dir, exist_ok=True)
        
        for row in selected_rows:
            frame_idx = int(row["frame_index"])
            x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            yolo_conf = float(row["yolo_conf"])
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Crop suspect bounding box (with simple safety guards)
                h, w, _ = frame.shape
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                crop = frame[y1_c:y2_c, x1_c:x2_c]
                
                if crop.size > 0:
                    crop = resize_and_pad(crop, 200, 250)
                    crop_filename = f"crop_frame_{frame_idx}_score_{yolo_conf:.2f}.jpg"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    time_sec = frame_idx / fps
                    top_frames.append({
                        "frame_index": frame_idx,
                        "time_sec": time_sec,
                        "yolo_conf": yolo_conf,
                        "crop_path": os.path.abspath(crop_path)
                    })
        cap.release()
        return top_frames
    except Exception as e:
        print("Error extracting top frames:", e)
        return []

def extract_single_candidate_crop(job_dir, vid_name, pid, rep_frame):
    import pandas as pd
    import cv2
    
    original_video = os.path.join(job_dir, "inputs", vid_name)
    output_dir = os.path.join(job_dir, "outputs")
    crops_dir = os.path.join(output_dir, "candidate_crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    crop_filename = f"candidate_p{pid}_frame_{rep_frame}.jpg"
    crop_path = os.path.join(crops_dir, crop_filename)
    if os.path.exists(crop_path):
        try:
            img = cv2.imread(crop_path)
            if img is not None:
                h_img, w_img = img.shape[:2]
                if w_img != 200 or h_img != 250:
                    padded = resize_and_pad(img, 200, 250)
                    cv2.imwrite(crop_path, padded)
        except Exception:
            pass
        return os.path.abspath(crop_path)
        
    csv_path = None
    for root, _, files in os.walk(output_dir):
        if "tracked_persons.csv" in files:
            csv_path = os.path.join(root, "tracked_persons.csv")
            break
            
    if not csv_path or not os.path.exists(csv_path) or not os.path.exists(original_video):
        return None
        
    try:
        df = pd.read_csv(csv_path)
        rows = df[(df["track_id"] == pid) & (df["frame_index"] == rep_frame)]
        if rows.empty:
            rows = df[df["track_id"] == pid]
            if rows.empty:
                return None
            bx = rows.iloc[0]
            rep_frame = int(bx["frame_index"])
        else:
            bx = rows.iloc[0]
            
        x1, y1, x2, y2 = int(bx['x1']), int(bx['y1']), int(bx['x2']), int(bx['y2'])
        
        cap = cv2.VideoCapture(original_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, rep_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            h, w, _ = frame.shape
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w, x2), min(h, y2)
            crop = frame[y1_c:y2_c, x1_c:x2_c]
            if crop.size > 0:
                crop = resize_and_pad(crop, 200, 250)
                cv2.imwrite(crop_path, crop)
                return os.path.abspath(crop_path)
    except Exception as e:
        print(f"Error extracting candidate crop for pid {pid}:", e)
    return None

def local_get_optimized_intervals(csv_path, pids, min_score_threshold=0.3):
    import pandas as pd
    from scipy.signal import find_peaks
    
    try:
        track_df = pd.read_csv(csv_path)
        person_df = track_df[track_df['track_id'].isin(pids)].copy()
        if person_df.empty: return []
        person_df = person_df.sort_values('frame_index').drop_duplicates(subset=['frame_index']).reset_index(drop=True)
        
        max_lap = track_df['laplacian'].max() if track_df['laplacian'].max() > 0 else 1.0
        person_df['saliency'] = (person_df['yolo_conf'] * 0.6) + ((person_df['laplacian'] / max_lap) * 0.4)
        person_df['smoothed'] = person_df['saliency'].rolling(window=15, min_periods=1, center=True).mean()
        
        saliency_array = person_df['smoothed'].fillna(0).values
        peaks, _ = find_peaks(saliency_array, height=min_score_threshold, distance=30)
        
        intervals = []
        for p in peaks:
            start_idx, end_idx = p, p
            while start_idx > 0 and saliency_array[start_idx] > (min_score_threshold * 0.5): start_idx -= 1
            while end_idx < len(saliency_array) - 1 and saliency_array[end_idx] > (min_score_threshold * 0.5): end_idx += 1
            intervals.append((int(person_df.loc[start_idx, 'frame_index']), int(person_df.loc[end_idx, 'frame_index'])))
            
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]] if intervals else []
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1] + 15: merged[-1] = (last[0], max(last[1], current[1]))
            else: merged.append(current)
        return merged
    except Exception as e:
        print("Error getting optimized intervals:", e)
        return []

def local_generate_evidence(job_dir, vid_name, pid, rep_frame, fps):
    import pandas as pd
    import cv2
    
    original_video = os.path.join(job_dir, "inputs", vid_name)
    output_dir = os.path.join(job_dir, "outputs")
    
    csv_path = None
    for root, _, files in os.walk(output_dir):
        if "tracked_persons.csv" in files:
            csv_path = os.path.join(root, "tracked_persons.csv")
            break
            
    if not csv_path or not os.path.exists(csv_path) or not os.path.exists(original_video):
        return None
        
    try:
        track_df = pd.read_csv(csv_path)
        person_tracks = track_df[track_df["track_id"] == pid]
        
        cap = cv2.VideoCapture(original_video)
        start_f = max(0, int(rep_frame - (fps * 3.0)))
        end_f = int(rep_frame + (fps * 6.0))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        ret, sample = cap.read()
        if not ret:
            cap.release()
            return None
            
        local_out_dir = os.path.join(output_dir, "local_gen", f"person_{pid}")
        os.makedirs(local_out_dir, exist_ok=True)
        
        out_path = os.path.join(local_out_dir, f"evidence_raw.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (sample.shape[1], sample.shape[0]))
        
        last_box = None
        template_img = None
        
        for f in range(start_f, end_f + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret: continue
            
            b_row = person_tracks[person_tracks["frame_index"] == f]
            if len(b_row) > 0:
                bx = b_row.iloc[0]
                x1, y1, x2, y2 = int(bx['x1']), int(bx['y1']), int(bx['x2']), int(bx['y2'])
                last_box = (x1, y1, x2, y2)
                
                template_img = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)].copy()
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"Target: ID {pid}", (x1, max(0, y1-10)), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                if template_img is not None and template_img.size > 0 and last_box is not None:
                    x1, y1, x2, y2 = last_box
                    pad = 50                 
                    sx1 = max(0, x1 - pad)
                    sy1 = max(0, y1 - pad)
                    sx2 = min(frame.shape[1], x2 + pad)
                    sy2 = min(frame.shape[0], y2 + pad)
                    search_region = frame[sy1:sy2, sx1:sx2]
                    
                    if search_region.shape[0] >= template_img.shape[0] and search_region.shape[1] >= template_img.shape[1]:
                        res = cv2.matchTemplate(search_region, template_img, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        
                        if max_val > 0.35: 
                            nx1 = sx1 + max_loc[0]
                            ny1 = sy1 + max_loc[1]
                            nx2 = nx1 + template_img.shape[1]
                            ny2 = ny1 + template_img.shape[0]
                            
                            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 165, 255), 3) 
                            cv2.putText(frame, "Target", (nx1, max(0, ny1-10)), \
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            
                            last_box = (nx1, ny1, nx2, ny2)
            
            time_sec = f / fps
            cv2.putText(frame, f"Time: {time_sec:.2f}s | Golden Frame: {rep_frame}", (10, 30), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)
                
        out.release()
        cap.release()
        return out_path
    except Exception as e:
        print("Error generating local evidence:", e)
        return None

def local_generate_synopsis(job_dir, vid_name, pids, intervals, fps, output_subdir_name):
    import pandas as pd
    import cv2
    
    original_video = os.path.join(job_dir, "inputs", vid_name)
    output_dir = os.path.join(job_dir, "outputs")
    
    csv_path = None
    for root, _, files in os.walk(output_dir):
        if "tracked_persons.csv" in files:
            csv_path = os.path.join(root, "tracked_persons.csv")
            break
            
    if not csv_path or not os.path.exists(csv_path) or not os.path.exists(original_video):
        return None
        
    try:
        track_df = pd.read_csv(csv_path)
        cap = cv2.VideoCapture(original_video)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        local_out_dir = os.path.join(output_dir, "local_gen", output_subdir_name)
        os.makedirs(local_out_dir, exist_ok=True)
        
        out_path = os.path.join(local_out_dir, f"synopsis_raw.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        
        person_tracks = track_df[track_df["track_id"].isin(pids)]
        pids_label = ",".join(str(p) for p in pids)
        
        for start_f, end_f in sorted(intervals):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            for f in range(start_f, end_f + 1):
                ret, frame = cap.read()
                if not ret: break
                
                b_rows = person_tracks[person_tracks["frame_index"] == f]
                for _, bx in b_rows.iterrows():
                    curr_pid = int(bx['track_id'])
                    x1, y1, x2, y2 = int(bx['x1']), int(bx['y1']), int(bx['x2']), int(bx['y2'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"Target: ID {curr_pid}", (x1, max(0, y1-10)), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, f"IDs: {pids_label} | Time: {f/fps:.2f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
                
        out.release()
        cap.release()
        return out_path
    except Exception as e:
        print("Error generating local synopsis:", e)
        return None

def local_transcode_audio_synopsis(original_video, synopsis_video, intervals, fps, output_path):
    import subprocess
    try:
        filter_parts = []
        for idx, (start_f, end_f) in enumerate(intervals):
            t_start = start_f / fps
            t_end = end_f / fps
            filter_parts.append(f"[0:a]atrim=start={t_start}:end={t_end},asetpts=PTS-STARTPTS[a{idx}]")
        concat_inputs = "".join(f"[a{k}]" for k in range(len(intervals)))
        filter_parts.append(f"{concat_inputs}concat=n={len(intervals)}:v=0:a=1[aud]")
        filter_graph = "; ".join(filter_parts)
        
        subprocess.run([
            "ffmpeg", "-y", "-i", original_video, "-i", synopsis_video,
            "-filter_complex", filter_graph,
            "-map", "1:v", "-map", "[aud]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", output_path
        ], check=True, capture_output=True)
        return True
    except Exception as e:
        print("Error transcoding local synopsis audio:", e)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", synopsis_video,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-map", "0:v?", "-map", "0:a?",
                output_path
            ], check=True, capture_output=True)
            return True
        except Exception:
            return False

def local_transcode_audio_evidence(original_video, evidence_video, rep_frame, fps, output_path):
    import subprocess
    try:
        start_f = max(0, int(rep_frame - (fps * 3.0)))
        t_start = start_f / fps
        t_end = t_start + 9.0
        filter_graph = f"[0:a]atrim=start={t_start}:end={t_end},asetpts=PTS-STARTPTS[aud]"
        
        subprocess.run([
            "ffmpeg", "-y", "-i", original_video, "-i", evidence_video,
            "-filter_complex", filter_graph,
            "-map", "1:v", "-map", "[aud]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", output_path
        ], check=True, capture_output=True)
        return True
    except Exception as e:
        print("Error transcoding local evidence audio:", e)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", evidence_video,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-map", "0:v?", "-map", "0:a?",
                output_path
            ], check=True, capture_output=True)
            return True
        except Exception:
            return False

@st.cache_resource
def get_job_queue():
    q = queue.Queue()
    job_statuses = {}
    return q, job_statuses

job_queue, job_statuses = get_job_queue()

@st.cache_resource
def start_worker_thread():
    def worker():
        while True:
            job_id, video_bytes, img_bytes, vid_name, img_name, selected_model = job_queue.get()
            job_statuses[job_id]['status'] = "Processing on Kaggle GPU..."
            job_statuses[job_id]['progress'] = 0.1
            
            try:
                # Map selection to directory and kernel ID
                if selected_model == "Model 1 (DeepFace)":
                    model_dir = "model1"
                    kernel_id = "directorprince/forensic-model-1"
                elif selected_model == "Model 2 (Best Retrieval)":
                    model_dir = "model2"
                    kernel_id = "directorprince/forensic-model-2"
                else:
                    model_dir = "model3"
                    kernel_id = "directorprince/forensic-video-retrieval-gpu"
                
                # Setup separate, persistent job directory for inputs and outputs
                job_dir = os.path.join("jobs", f"job_{job_id}")
                input_copy_dir = os.path.join(job_dir, "inputs")
                os.makedirs(input_copy_dir, exist_ok=True)
                
                # We always upload inputs to a shared dataset using model3's input_data folder 
                # (since it's configured for 'directorprince/forensic-input')
                input_dir = os.path.join("model3", "input_data")
                os.makedirs(input_dir, exist_ok=True)
                
                # Clear existing files in the shared input directory
                for f in os.listdir(input_dir):
                    if f != "dataset-metadata.json":
                        os.remove(os.path.join(input_dir, f))
                        
                vid_path = os.path.join(input_dir, vid_name)
                img_path = os.path.join(input_dir, img_name)
                
                # Write to shared folder (for Kaggle datasets update)
                with open(vid_path, "wb") as f:
                    f.write(video_bytes)
                if img_bytes:
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                
                # Save persistent copies to the job-specific inputs directory
                with open(os.path.join(input_copy_dir, vid_name), "wb") as f:
                    f.write(video_bytes)
                if img_bytes:
                    with open(os.path.join(input_copy_dir, img_name), "wb") as f:
                        f.write(img_bytes)
                    
                job_statuses[job_id]['status'] = "Pushing data to Kaggle..."
                job_statuses[job_id]['progress'] = 0.2
                
                try:
                    subprocess.run([sys.executable, "-m", "kaggle", "datasets", "version", "-p", input_dir, "-m", f"Update for job {job_id}"], check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    try:
                        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "create", "-p", input_dir], check=True, capture_output=True)
                    except subprocess.CalledProcessError:
                        raise e

                job_statuses[job_id]['status'] = f"Triggering Kaggle Kernel ({model_dir})..."
                job_statuses[job_id]['progress'] = 0.3
                
                # Trigger specific Kernel
                subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", model_dir], check=True, capture_output=True)
                
                # Poll Kernel and direct outputs to job-specific directory
                OUTPUT_DIR = os.path.join(job_dir, "outputs")
                start_time = time.time()
                while True:
                    res = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
                    status = res.stdout.strip()
                    
                    elapsed = int(time.time() - start_time)
                    job_statuses[job_id]['status'] = f"Elapsed: {elapsed}s | Status: {status}"
                    job_statuses[job_id]['progress'] = min(0.3 + (elapsed / 500), 0.95)
                    
                    if "complete" in status.lower() or "error" in status.lower():
                        break
                    time.sleep(15)
                
                job_statuses[job_id]['status'] = "Downloading results..."
                job_statuses[job_id]['progress'] = 0.97
                
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", kernel_id, "-p", OUTPUT_DIR], check=False, capture_output=True)
                
                # Find output videos (could be .mp4 or .avi depending on the model)
                output_videos = []
                for root, _, files in os.walk(OUTPUT_DIR):
                    for f in files:
                        f_lower = f.lower()
                        if '_playable_' in f_lower or 'ffmpeg_test' in f_lower:
                            continue
                        if (f.endswith('.mp4') or f.endswith('.avi')) and ('evidence' in f_lower or 'output' in f_lower or 'synopsis' in f_lower):
                            output_videos.append(os.path.join(root, f))
                
                if output_videos:
                    # Sort: prioritize "synopsis" first, then "output", then "evidence"
                    def sort_key(filepath):
                        filename = os.path.basename(filepath).lower()
                        if 'synopsis' in filename:
                            return 0
                        if 'output' in filename:
                            return 1
                        return 2
                    output_videos.sort(key=sort_key)
                    
                    playable_paths = []
                    
                    # Try to load job metadata for audio synchronization (searching recursively)
                    job_meta = {}
                    meta_path = None
                    for root, _, files in os.walk(OUTPUT_DIR):
                        if "job_meta.json" in files:
                            meta_path = os.path.join(root, "job_meta.json")
                            break
                    if meta_path and os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r") as fm:
                                job_meta = json.load(fm)
                        except Exception:
                            pass
                            
                    # Check if original video has audio
                    original_video = os.path.join(job_dir, "inputs", vid_name)
                    has_audio = False
                    if os.path.exists(original_video):
                        try:
                            res = subprocess.run([
                                "ffprobe", "-v", "error", "-select_streams", "a", 
                                "-show_entries", "stream=codec_name", "-of", "csv=p=0", 
                                original_video
                            ], capture_output=True, text=True, check=True)
                            if res.stdout.strip():
                                has_audio = True
                        except Exception:
                            pass
                            
                    for i, vid in enumerate(output_videos):
                        filename_base = os.path.basename(vid)
                        clean_name = filename_base.replace(".mp4", "").replace(".avi", "")
                        playable_mp4 = os.path.join(OUTPUT_DIR, f"{clean_name}_playable_{job_id}_{i}.mp4")
                        
                        transcode_success = False
                        
                        # Try to transcode using FFmpeg with audio extraction from original video if possible
                        if has_audio and job_meta:
                            try:
                                fps = job_meta.get("fps", 30.0)
                                if "synopsis" in filename_base.lower():
                                    intervals = job_meta.get("intervals", [])
                                    if intervals:
                                        # Construct filter graph for multi-interval audio concatenation
                                        filter_parts = []
                                        for idx, (start_f, end_f) in enumerate(intervals):
                                            t_start = start_f / fps
                                            t_end = end_f / fps
                                            filter_parts.append(f"[0:a]atrim=start={t_start}:end={t_end},asetpts=PTS-STARTPTS[a{idx}]")
                                        concat_inputs = "".join(f"[a{k}]" for k in range(len(intervals)))
                                        filter_parts.append(f"{concat_inputs}concat=n={len(intervals)}:v=0:a=1[aud]")
                                        filter_graph = "; ".join(filter_parts)
                                        
                                        subprocess.run([
                                            "ffmpeg", "-y", "-i", original_video, "-i", vid,
                                            "-filter_complex", filter_graph,
                                            "-map", "1:v", "-map", "[aud]",
                                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                            "-c:a", "aac", playable_mp4
                                        ], check=True, capture_output=True)
                                        playable_paths.append(playable_mp4)
                                        transcode_success = True
                                        
                                elif "evidence" in filename_base.lower():
                                    rep_frame = job_meta.get("rep_frame")
                                    if rep_frame is not None:
                                        start_f = max(0, int(rep_frame - (fps * 3.0)))
                                        t_start = start_f / fps
                                        t_end = t_start + 9.0  # evidence clip is 9.0 seconds long (3s before, 6s after)
                                        filter_graph = f"[0:a]atrim=start={t_start}:end={t_end},asetpts=PTS-STARTPTS[aud]"
                                        
                                        subprocess.run([
                                            "ffmpeg", "-y", "-i", original_video, "-i", vid,
                                            "-filter_complex", filter_graph,
                                            "-map", "1:v", "-map", "[aud]",
                                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                            "-c:a", "aac", playable_mp4
                                        ], check=True, capture_output=True)
                                        playable_paths.append(playable_mp4)
                                        transcode_success = True
                            except Exception:
                                pass
                                
                        # Fallback to direct FFmpeg transcode (no audio sync)
                        if not transcode_success:
                            try:
                                subprocess.run([
                                    "ffmpeg", "-y", "-i", vid,
                                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                    "-c:a", "aac", "-map", "0:v?", "-map", "0:a?",
                                    playable_mp4
                                ], check=True, capture_output=True)
                                playable_paths.append(playable_mp4)
                                transcode_success = True
                            except Exception:
                                pass
                        
                        # Fallback to moviepy
                        if not transcode_success:
                            try:
                                from moviepy import VideoFileClip
                                clip = VideoFileClip(vid)
                                clip.write_videofile(playable_mp4, codec="libx264", audio_codec="aac", logger=None)
                                playable_paths.append(playable_mp4)
                                transcode_success = True
                            except Exception:
                                pass
                        
                        # Absolute fallback
                        if not transcode_success:
                            playable_paths.append(vid)
                    
                    if playable_paths:
                        job_statuses[job_id]['result_paths'] = playable_paths
                        job_statuses[job_id]['result_path'] = playable_paths[0]  # backward compatibility
                        
                        # Pre-extract top 10 matching frames of target
                        target_id = job_meta.get("target_id")
                        if target_id is not None:
                            extract_top_frames(job_dir, vid_name, target_id)
                    else:
                        job_statuses[job_id]['error'] = "Failed to make any output video playable."
                else:
                    job_statuses[job_id]['error'] = "Output video not found in Kaggle results."

                job_statuses[job_id]['status'] = "Done!"
                job_statuses[job_id]['progress'] = 1.0
                job_statuses[job_id]['completed'] = True
                
            except subprocess.CalledProcessError as e:
                job_statuses[job_id]['status'] = "Failed!"
                err_msg = str(e)
                if e.stderr: err_msg += f" | STDERR: {e.stderr.decode('utf-8', errors='ignore').strip()}"
                if e.stdout: err_msg += f" | STDOUT: {e.stdout.decode('utf-8', errors='ignore').strip()}"
                job_statuses[job_id]['error'] = err_msg
                job_statuses[job_id]['completed'] = True
            except Exception as e:
                job_statuses[job_id]['status'] = "Failed!"
                job_statuses[job_id]['error'] = str(e)
                job_statuses[job_id]['completed'] = True
                
            job_queue.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t

start_worker_thread()

# --- UI IMPLEMENTATION ---
st.set_page_config(page_title="Master Controller | Forensic Video Retrieval", page_icon="🕵️", layout="wide")
st.title("🕵️ Forensic Master Controller")
st.markdown("Run multiple totally different AI models on Kaggle GPU from one central interface.")

# --- CUSTOM CSS FOR BEAUTIFUL GRID, CARDS & COMPACT VIDEOS ---
st.markdown(
    """
    <style>
    /* Styling for image elements */
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s cubic-bezier(0.165, 0.84, 0.44, 1), box-shadow 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stImage > img:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    /* Make captions match nicely */
    .stImage > div > p {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        margin-top: 4px !important;
        color: #ddd !important;
        text-align: center !important;
    }
    /* Limit height of video players to make them less overpowering */
    div[data-testid="stVideo"] video {
        max-height: 380px !important;
        width: auto !important;
        max-width: 100% !important;
        margin: 0 auto !important;
        display: block !important;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

if 'job_id' not in st.session_state:
    st.session_state.job_id = None

selected_model = st.sidebar.selectbox("Choose AI Model", [
    "Model 3 (YOLOv8 + EffNet + Indexing)",
    "Model 1 (DeepFace)", 
    "Model 2 (Best Retrieval)"
])

# --- History / Load Past Job ---
if os.path.exists("jobs"):
    past_jobs = [d.replace("job_", "") for d in os.listdir("jobs") if d.startswith("job_")]
    if past_jobs:
        selected_past_job = st.sidebar.selectbox("📂 Load Past Job Results", ["Select a job..."] + sorted(past_jobs, reverse=True))
        if selected_past_job != "Select a job...":
            st.session_state.job_id = selected_past_job

col1, col2 = st.columns(2)
with col1:
    uploaded_video = st.file_uploader("1. Upload Evidence Video", type=['mp4'])
with col2:
    uploaded_image = st.file_uploader("2. Upload Query Image (Optional for Model 2)", type=['jpeg', 'jpg', 'png'])

if st.button("Generate Forensic Synopsis", type="primary"):
    if not uploaded_video:
        st.error("Please upload an Evidence Video!")
    else:
        new_job_id = str(uuid4())[:8]
        st.session_state.job_id = new_job_id
        position = job_queue.qsize() + 1
        
        job_statuses[new_job_id] = {
            'status': f"Waiting in Queue... You are position #{position}",
            'progress': 0.0,
            'completed': False,
            'result_path': None,
            'error': None
        }
        
        img_bytes = uploaded_image.getbuffer() if uploaded_image else None
        img_name = uploaded_image.name if uploaded_image else "auto_query.jpg"
        
        job_queue.put((
            new_job_id, 
            uploaded_video.getbuffer(), 
            img_bytes,
            uploaded_video.name,
            img_name,
            selected_model
        ))

# Reconstruct job status from disk if not in memory (persistency layer)
if st.session_state.job_id and st.session_state.job_id not in job_statuses:
    job_dir = os.path.join("jobs", f"job_{st.session_state.job_id}")
    if os.path.exists(job_dir):
        playable_videos = []
        outputs_dir = os.path.join(job_dir, "outputs")
        if os.path.exists(outputs_dir):
            for root, _, files in os.walk(outputs_dir):
                for f in files:
                    if f.endswith(".mp4") and "_playable_" in f:
                        playable_videos.append(os.path.join(root, f))
                        
        if playable_videos:
            playable_videos.sort(key=lambda x: 0 if "synopsis" in os.path.basename(x).lower() else 1)
            job_statuses[st.session_state.job_id] = {
                'status': "Loaded from history",
                'progress': 1.0,
                'completed': True,
                'result_paths': playable_videos,
                'result_path': playable_videos[0],
                'error': None
            }

if st.session_state.job_id:
    job = job_statuses.get(st.session_state.job_id)
    if job:
        st.write("---")
        st.subheader("Job Status")
        st.info(job['status'])
        st.progress(job['progress'])
        
        if job['completed']:
            if job['error']:
                st.error(f"Job failed: {job['error']}")
            elif job.get('result_paths'):
                st.success("Synopsis Generated Successfully!")
                
                # Locate job directories
                job_dir = os.path.join("jobs", f"job_{st.session_state.job_id}")
                inputs_dir = os.path.join(job_dir, "inputs")
                
                # Find original video name
                v_name = None
                if os.path.exists(inputs_dir):
                    for f in os.listdir(inputs_dir):
                        if f.endswith(".mp4"):
                            v_name = f
                            break
                            
                # 1. Display suspect query image at the top (centered)
                query_img_path = None
                if os.path.exists(inputs_dir):
                    for f in os.listdir(inputs_dir):
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            query_img_path = os.path.join(inputs_dir, f)
                            break
                            
                if query_img_path and os.path.exists(query_img_path):
                    _, img_mid_col, _ = st.columns([1.5, 2, 1.5])
                    with img_mid_col:
                        st.markdown("### 🎯 Suspect Query Image")
                        st.image(query_img_path, use_container_width=True)
                        
                # Retrieve target_id by parsing job_meta.json
                target_id = None
                job_meta = {}
                meta_path = None
                for root, _, files in os.walk(os.path.join(job_dir, "outputs")):
                    if "job_meta.json" in files:
                        meta_path = os.path.join(root, "job_meta.json")
                        break
                if meta_path and os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as fm:
                            job_meta = json.load(fm)
                            target_id = job_meta.get("target_id")
                    except Exception:
                        pass
                
                fps = job_meta.get("fps", 30.0)
                top_matches = job_meta.get("top_matches", [])
                
                # If we have top_matches list (new interactive feedback flow)
                if top_matches and v_name:
                    st.write("---")
                    st.subheader("🎯 Top Detected Suspect Matches")
                    st.info("Below are the top matched candidates detected in the video. Review their faces and select the correct target to generate their synopsis.")
                    
                    # Display the candidates in a grid
                    match_cols = st.columns(5)
                    for k, match in enumerate(top_matches[:10]):
                        with match_cols[k % 5]:
                            pid = match["person_id"]
                            score = match["final_score"]
                            rep_f = match["rep_frame"]
                            
                            # Extract crop
                            crop_file = extract_single_candidate_crop(job_dir, v_name, pid, rep_f)
                            if crop_file and os.path.exists(crop_file):
                                caption = f"Person ID: {pid} | Score: {score:.2f}"
                                st.image(crop_file, caption=caption, use_container_width=True)
                                
                    # Multi-selection dropdown
                    st.write("")
                    candidate_ids = [m["person_id"] for m in top_matches[:10]]
                    
                    state_key = f"selected_pids_{st.session_state.job_id}"
                    if state_key not in st.session_state:
                        st.session_state[state_key] = [target_id]
                        
                    selected_pids = st.multiselect(
                        "🔍 Select one or more Person IDs to generate their merged synopsis:",
                        candidate_ids,
                        default=st.session_state[state_key],
                        key=f"multiselect_{st.session_state.job_id}"
                    )
                    
                    # Sort selected pids to keep cache key consistent
                    selected_pids = sorted(selected_pids)
                    
                    if selected_pids != st.session_state[state_key]:
                        # Make sure at least one is selected, else fallback to target_id
                        if not selected_pids:
                            st.session_state[state_key] = [target_id]
                        else:
                            st.session_state[state_key] = selected_pids
                        st.rerun()
                        
                    if selected_pids:
                        pids_str = "_".join(str(p) for p in selected_pids)
                        local_person_dir = os.path.join(job_dir, "outputs", "local_gen", f"group_{pids_str}")
                        os.makedirs(local_person_dir, exist_ok=True)
                        
                        syn_playable = os.path.join(local_person_dir, "synopsis_playable.mp4")
                        
                        # Track evidence playables for each selected pid
                        ev_playables = {}
                        for pid in selected_pids:
                            ev_dir = os.path.join(job_dir, "outputs", "local_gen", f"person_{pid}")
                            os.makedirs(ev_dir, exist_ok=True)
                            ev_playables[pid] = os.path.join(ev_dir, "evidence_playable.mp4")
                            
                        # Check if generation is needed
                        needs_gen = not os.path.exists(syn_playable) or any(not os.path.exists(path) for path in ev_playables.values())
                        
                        if needs_gen:
                            pids_label = ", ".join(str(p) for p in selected_pids)
                            with st.spinner(f"⚡ Generating custom merged synopsis and evidence clips for Person IDs [{pids_label}] locally..."):
                                csv_path = None
                                for root, _, files in os.walk(os.path.join(job_dir, "outputs")):
                                    if "tracked_persons.csv" in files:
                                        csv_path = os.path.join(root, "tracked_persons.csv")
                                        break
                                
                                if csv_path:
                                    # 1. Generate merged synopsis
                                    if not os.path.exists(syn_playable):
                                        intervals = local_get_optimized_intervals(csv_path, selected_pids)
                                        if intervals:
                                            raw_syn = local_generate_synopsis(job_dir, v_name, selected_pids, intervals, fps, f"group_{pids_str}")
                                            if raw_syn:
                                                local_transcode_audio_synopsis(os.path.join(job_dir, "inputs", v_name), raw_syn, intervals, fps, syn_playable)
                                        else:
                                            st.warning(f"No high-saliency tracks found for Person IDs [{pids_label}].")
                                    
                                    # 2. Generate evidence clip for each PID if missing
                                    for pid in selected_pids:
                                        ev_playable = ev_playables[pid]
                                        if not os.path.exists(ev_playable):
                                            rep_f = next((m["rep_frame"] for m in top_matches if m["person_id"] == pid), int(job_meta.get("rep_frame", 0)))
                                            raw_ev = local_generate_evidence(job_dir, v_name, pid, rep_f, fps)
                                            if raw_ev:
                                                local_transcode_audio_evidence(os.path.join(job_dir, "inputs", v_name), raw_ev, rep_f, fps, ev_playable)
                                                
                        # Display the generated videos
                        st.write("---")
                        pids_label = ", ".join(str(p) for p in selected_pids)
                        st.subheader(f"🎬 Evidence Tracks & Merged Video Synopsis (Person IDs: {pids_label})")
                        
                        # 1. First, display individual evidence zoom tracks for each selected suspect
                        if len(selected_pids) == 1:
                            pid = selected_pids[0]
                            ev_playable = ev_playables[pid]
                            if os.path.exists(ev_playable):
                                _, mid_col, _ = st.columns([1.5, 3, 1.5])
                                with mid_col:
                                    st.markdown(f"#### 📹 Evidence Zoom Track (Person ID: {pid})")
                                    st.video(ev_playable)
                        else:
                            st.markdown("#### 📹 Individual Evidence Zoom Tracks")
                            ev_cols = st.columns(2)
                            ev_idx = 0
                            for pid in selected_pids:
                                ev_playable = ev_playables[pid]
                                if os.path.exists(ev_playable):
                                    with ev_cols[ev_idx % 2]:
                                        st.markdown(f"**Person ID: {pid}**")
                                        st.video(ev_playable)
                                    ev_idx += 1
                                    
                        # 2. Finally, display the merged video synopsis at the end
                        if os.path.exists(syn_playable):
                            _, mid_col, _ = st.columns([1.5, 3, 1.5])
                            with mid_col:
                                st.markdown(f"#### 📹 Merged Video Synopsis (Person IDs: {pids_label})")
                                st.video(syn_playable)
                                
                else:
                    # Fallback to displaying top matches of the target suspect (backward compatibility for old runs)
                    if target_id is not None:
                        st.write("---")
                        st.subheader("🎯 Top 10 Detected Matches (Suspect Crops)")
                        
                        crops_path = os.path.join(job_dir, "outputs", "top_crops")
                        top_frames = []
                        if os.path.exists(crops_path):
                            for f in os.listdir(crops_path):
                                if f.startswith("crop_frame_") and f.endswith(".jpg"):
                                    try:
                                        parts = f.replace(".jpg", "").split("_")
                                        f_idx = int(parts[2])
                                        score = float(parts[4])
                                        c_path = os.path.join(crops_path, f)
                                        # Check and resize/pad existing crop
                                        try:
                                            import cv2
                                            img = cv2.imread(c_path)
                                            if img is not None:
                                                h_img, w_img = img.shape[:2]
                                                if w_img != 200 or h_img != 250:
                                                    padded = resize_and_pad(img, 200, 250)
                                                    cv2.imwrite(c_path, padded)
                                        except Exception:
                                            pass
                                        top_frames.append({
                                            "frame_index": f_idx,
                                            "yolo_conf": score,
                                            "crop_path": c_path
                                        })
                                    except Exception:
                                        pass
                        
                        if not top_frames and v_name:
                            top_frames = extract_top_frames(job_dir, v_name, target_id)
                                
                        if top_frames:
                            top_frames = sorted(top_frames, key=lambda x: x["yolo_conf"], reverse=True)
                            img_cols = st.columns(5)
                            for k, frame in enumerate(top_frames[:10]):
                                with img_cols[k % 5]:
                                    caption = f"Frame {frame['frame_index']} | Conf: {frame['yolo_conf']:.2f}"
                                    st.image(frame['crop_path'], caption=caption, use_container_width=True)
                                    
                    st.write("---")
                    st.subheader("🎬 Video Synopsis & Evidence Tracks")
                    for idx, path in enumerate(job['result_paths']):
                        _, mid_col, _ = st.columns([1.5, 3, 1.5])
                        with mid_col:
                            base = os.path.basename(path)
                            label = base
                            if '_playable_' in base:
                                label = base.split('_playable_')[0]
                            label = label.replace('_', ' ').replace('-', ' ').title()
                            st.markdown(f"#### 📹 {label}")
                            st.video(path)
            elif job.get('result_path'):
                st.success("Synopsis Generated Successfully!")
                _, mid_col, _ = st.columns([1.5, 3, 1.5])
                with mid_col:
                    st.video(job['result_path'])
        else:
            time.sleep(2)
            st.rerun()

# --- SYSTEM WALKTHROUGH SECTION ---
st.write("---")
with st.expander("📖 System Walkthrough & How It Works"):
    st.markdown("""
    ### 📂 Structured Job Storage
    Every run generates a persistent job folder at `jobs/job_{job_id}/`:
    * **Inputs**: Stores original video and query image.
    * **Outputs**: Stores logs, CSV files, and transcoded playable videos.
    
    ### 🕵️ Bounding Box Tracking & Focus Saliency
    * **Model 3** tracks suspects in each frame and computes a Saliency score based on YOLO confidence and frame sharpness (Laplacian variance).
    * Peak detection is used to extract the key frames where the suspect is clearest and most visible.
    
    ### 🔊 Audio Synchronization
    * OpenCV's video writer strips audio during frame drawing.
    * This controller reads the metadata intervals and uses **FFmpeg** to extract and stitch together corresponding audio segments from the original video.
    
    ### 🎯 Top Matching Crops
    * The system extracts the top 10 frames with the highest confidence scores, crops the suspect's bounding boxes, and displays them as a match grid.
    """)
