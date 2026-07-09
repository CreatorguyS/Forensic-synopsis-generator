import json
import os

# --- CONFIGURE MODEL 1 ---
m1_path = r"c:\complete web development camp\project\video_retrieval\model1\main.ipynb"
with open(m1_path, "r", encoding="utf-8") as f:
    m1 = json.load(f)

# Find config cell in model 1
for cell in m1["cells"]:
    if cell["cell_type"] == "code" and "INPUT_PATH =" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if line.startswith("INPUT_PATH ="):
                new_source.append("import glob\n")
                new_source.append("try:\n")
                new_source.append("    INPUT_PATH = glob.glob('/kaggle/input/forensic-input/*.mp4')[0]\n")
                new_source.append("except:\n")
                new_source.append("    " + line)
            elif line.startswith("QUERY_IMAGE ="):
                new_source.append("try:\n")
                new_source.append("    QUERY_IMAGE = [f for f in glob.glob('/kaggle/input/forensic-input/*') if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')][0]\n")
                new_source.append("except:\n")
                new_source.append("    " + line)
            else:
                new_source.append(line)
        cell["source"] = new_source
        break

with open(m1_path, "w", encoding="utf-8") as f:
    json.dump(m1, f, indent=1)

m1_meta = {
  "id": "directorprince/forensic-model-1",
  "title": "Forensic Model 1",
  "code_file": "main.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": ["directorprince/forensic-input"],
  "competition_sources": [],
  "kernel_sources": [],
  "accelerator": "NVIDIA_TESLA_T4"
}
with open(r"c:\complete web development camp\project\video_retrieval\model1\kernel-metadata.json", "w") as f:
    json.dump(m1_meta, f, indent=2)

print("Model 1 configured.")

# --- CONFIGURE MODEL 2 ---
m2_path = r"c:\complete web development camp\project\video_retrieval\model2\best-retrieval.ipynb"
with open(m2_path, "r", encoding="utf-8") as f:
    m2 = json.load(f)

for cell in m2["cells"]:
    if cell["cell_type"] == "code" and "VIDEO_FOLDER =" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if line.startswith("VIDEO_FOLDER ="):
                new_source.append("VIDEO_FOLDER = '/kaggle/input/forensic-input'\n")
            elif line.startswith("QUERY_IMAGE ="):
                new_source.append("import glob\n")
                new_source.append("try:\n")
                new_source.append("    QUERY_IMAGE = [f for f in glob.glob('/kaggle/input/forensic-input/*') if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')][0]\n")
                new_source.append("except:\n")
                new_source.append("    " + line)
            else:
                new_source.append(line)
        cell["source"] = new_source
        break

with open(m2_path, "w", encoding="utf-8") as f:
    json.dump(m2, f, indent=1)

m2_meta = {
  "id": "directorprince/forensic-model-2",
  "title": "Forensic Model 2",
  "code_file": "best-retrieval.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": ["directorprince/forensic-input"],
  "competition_sources": [],
  "kernel_sources": [],
  "accelerator": "NVIDIA_TESLA_T4"
}
with open(r"c:\complete web development camp\project\video_retrieval\model2\kernel-metadata.json", "w") as f:
    json.dump(m2_meta, f, indent=2)

print("Model 2 configured.")
