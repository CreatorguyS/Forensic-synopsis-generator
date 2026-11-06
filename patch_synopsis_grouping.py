import json

filepath = 'model4/capstone final code.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    d = json.load(f)

for i, cell in enumerate(d['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    # Find Cell 11 (Optimization, Synopsis, and Metrics Module)
    if 'def get_optimized_intervals' in src and 'def create_final_synopsis' in src:
        new_src = '''import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# --- OPTIMIZATION, SYNOPSIS, AND METRICS MODULE ---
def get_optimized_intervals(track_df, person_ids, min_score_threshold=0.3):
    # Support both single integer ID and list/set of IDs
    if isinstance(person_ids, (int, float)):
        person_ids = [int(person_ids)]
    else:
        person_ids = [int(pid) for pid in person_ids]
        
    person_df = track_df[track_df['track_id'].isin(person_ids)].copy()
    if person_df.empty: return []
    person_df = person_df.sort_values('frame_index').reset_index(drop=True)
    
    if 'laplacian' not in track_df.columns: track_df['laplacian'] = 0.0
    max_lap = track_df['laplacian'].max() if track_df['laplacian'].max() > 0 else 1.0
    person_df['laplacian'] = person_df.get('laplacian', 0.0)
    person_df['saliency'] = (person_df['yolo_conf'] * 0.6) + ((person_df['laplacian'] / max_lap) * 0.4)
    person_df['smoothed'] = person_df['saliency'].rolling(window=15, min_periods=1, center=True).mean()
    
    saliency_array = person_df['smoothed'].fillna(0).values
    peaks, _ = find_peaks(saliency_array, height=min_score_threshold, distance=30)
    
    intervals = []
    for p in peaks:
        start_idx, end_idx = p, p
        while start_idx > 0 and saliency_array[start_idx] > (min_score_threshold * 0.5): start_idx -= 1
        while end_idx < len(saliency_array) - 1 and saliency_array[end_idx] > (min_score_threshold * 0.5): end_idx += 1
        # Safe boundary checks
        if start_idx < len(person_df) and end_idx < len(person_df):
            intervals.append((int(person_df.loc[start_idx, 'frame_index']), int(person_df.loc[end_idx, 'frame_index'])))
        
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]] if intervals else []
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 15: merged[-1] = (last[0], max(last[1], current[1]))
        else: merged.append(current)
    return merged

def create_final_synopsis(optimized_intervals, video_path, person_ids, output_dir, track_df):
    if isinstance(person_ids, (int, float)):
        person_ids = [int(person_ids)]
    else:
        person_ids = [int(pid) for pid in person_ids]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(output_dir, "final_synopsis.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    # Filter track data for target persons once to speed up lookups
    person_tracks = track_df[track_df["track_id"].isin(person_ids)]
    
    for start_f, end_f in sorted(optimized_intervals):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for f in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret: break
            
            # Find and draw bounding box for this frame
            b_rows = person_tracks[person_tracks["frame_index"] == f]
            if len(b_rows) > 0:
                bx = b_rows.sort_values("yolo_conf", ascending=False).iloc[0]
                x1, y1, x2, y2 = int(bx['x1']), int(bx['y1']), int(bx['x2']), int(bx['y2'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"Target: ID {bx['track_id']}", (x1, max(0, y1-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            primary_id = person_ids[0]
            cv2.putText(frame, f"ID: {primary_id} | Time: {f/fps:.2f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)
    out.release(); cap.release()
    return out_path

def calculate_evaluation_metrics(orig_path, syn_path, gt_events, intervals):
    cap_o = cv2.VideoCapture(orig_path); cap_s = cv2.VideoCapture(syn_path)
    fps = cap_o.get(cv2.CAP_PROP_FPS) or 25.0
    s_frames = cap_s.get(cv2.CAP_PROP_FRAME_COUNT)
    o_frames = cap_o.get(cv2.CAP_PROP_FRAME_COUNT)
    cr = o_frames / s_frames if s_frames > 0 else 1.0
    cap_o.release(); cap_s.release()
    captured_seconds = [(s/fps, e/fps) for s, e in intervals]
    recall = sum(1 for gt in gt_events if any(c_s <= gt[1] and c_e >= gt[0] for c_s, c_e in captured_seconds)) / len(gt_events)
    return {"Compression Ratio": round(cr, 2), "Event Recall (%)": round(recall * 100, 2)}

# --- EXECUTION ---
best_match = best_match_result
track_df = pd.read_csv(os.path.join(DIRS["track"], "tracked_persons.csv"))

# Smart ID grouping: include all track IDs representing the same person (similarity score within 85% of best match)
matched_person_ids = [best_match['person_id']]
try:
    best_score = best_match['final_score']
    # results is global, holds candidate scores
    for r in results:
        pid = r['person_id']
        score = r['final_score']
        if pid != best_match['person_id'] and score >= best_score * 0.82 and score > 0.25:
            matched_person_ids.append(pid)
    print(f"Grouped matching suspect Track IDs representing the same identity: {matched_person_ids}")
except Exception as e:
    print(f"Suspect grouping failed: {e}. Falling back to single best match ID.")

intervals = get_optimized_intervals(track_df, matched_person_ids)
if len(intervals) == 0:
    print('No highly salient intervals found for synopsis. Skipping synopsis generation.')
    syn_path = 'N/A'
    metrics = {'Compression Ratio': 0.0, 'Event Recall (%)': 0.0}
else:
    syn_path = create_final_synopsis(intervals, VIDEO_PATH, matched_person_ids, DIRS["output"], track_df)
    metrics = calculate_evaluation_metrics(VIDEO_PATH, syn_path, [(10.0, 15.0), (30.0, 40.0)], intervals)

# Save job metadata for audio reconstruction on local machine
import json
top_matches_serializable = []
for res in results[:10]:
    top_matches_serializable.append({
        "person_id": int(res["person_id"]),
        "rep_frame": int(res["rep_frame"]),
        "final_score": float(res["final_score"]),
        "weights": [float(w) for w in res["weights"]]
    })

job_meta = {
    "target_id": int(best_match['person_id']),
    "rep_frame": int(best_match['rep_frame']),
    "fps": float(cv2.VideoCapture(VIDEO_PATH).get(cv2.CAP_PROP_FPS) or 25.0),
    "intervals": [[int(start), int(end)] for start, end in intervals] if intervals else [],
    "top_matches": top_matches_serializable
}
with open(os.path.join(DIRS["output"], "job_meta.json"), "w") as f:
    json.dump(job_meta, f)

print(f"\n--- REPORT ---\n{metrics}\nSynopsis saved: {syn_path}")
'''
        cell['source'] = [new_src]
        print(f"Cell {i}: Upgraded synopsis generation to support grouping of multiple matched track IDs representing the same identity")
        break

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)

print("Saved model4")
