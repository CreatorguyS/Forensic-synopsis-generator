import os
from moviepy import VideoFileClip

input_path = r"c:\complete web development camp\project\video_retrieval\model3\final_results\forensic_pipeline\final_output\evidence_failover_p3.mp4"
output_path = r"c:\complete web development camp\project\video_retrieval\model3\final_results\forensic_pipeline\final_output\evidence_playable.mp4"

print(f"Converting video to H.264...")
try:
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"\nSuccess! New video saved at: {output_path}")
except Exception as e:
    print(f"Error during conversion: {e}")
