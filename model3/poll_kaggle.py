import time
import subprocess
import os

KERNEL = "directorprince/forensic-video-retrieval-gpu"
POLL_INTERVAL = 30 
OUTPUT_DIR = "kaggle_output"

print(f"Polling kernel: {KERNEL}")
print(f"Output dir: {OUTPUT_DIR}")
print("-" * 60)

start_time = time.time()

while True:
    try:
        # Check status
        result = subprocess.run(
            ["python", "-m", "kaggle", "kernels", "status", KERNEL],
            capture_output=True, text=True, check=True
        )
        status_line = result.stdout.strip()
        
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        print(f"[{mins:02d}:{secs:02d}] {status_line}")
        
        if "complete" in status_line.lower():
            print("\nKernel finished successfully! Downloading output...")
            subprocess.run(["python", "-m", "kaggle", "kernels", "output", KERNEL, "-p", OUTPUT_DIR])
            break
            
        elif "error" in status_line.lower():
            print("\nKernel encountered an error! Downloading logs...")
            subprocess.run(["python", "-m", "kaggle", "kernels", "output", KERNEL, "-p", OUTPUT_DIR])
            break
            
        elif "cancel" in status_line.lower():
            print("\nKernel was cancelled.")
            break
            
    except subprocess.CalledProcessError as e:
        print(f"Error checking status: {e.stderr}")
        
    time.sleep(POLL_INTERVAL)
