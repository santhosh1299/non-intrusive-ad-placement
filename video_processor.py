import cv2
import os
import random
import shutil
from moviepy.editor import ImageSequenceClip
import numpy as np

# --- Scene Detection (using pixel difference and histogram comparison) ---
def detect_scenes(video_path, scene_threshold=0.7, min_scene_length=30):
    cap = cv2.VideoCapture(video_path)
    scenes = []
    prev_frame = None
    scene_start = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray_frame)
            non_zero = np.count_nonzero(diff)

            hist_prev = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
            hist_curr = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)

            motion_score = max(non_zero / (frame.size), hist_diff)

            if motion_score > scene_threshold:
                if frame_count - scene_start > min_scene_length:
                    scenes.append((scene_start, frame_count - 1))
                scene_start = frame_count

        prev_frame = gray_frame
        frame_count += 1

    if frame_count - scene_start > min_scene_length:
        scenes.append((scene_start, frame_count - 1))

    cap.release()
    return scenes

def overlay_logo_on_frame(frame, logo):
    height, width = frame.shape[:2]
    logo_resized = cv2.resize(logo, (int(width * 0.2), int(logo.shape[0] * (width * 0.2) / logo.shape[1])))
   
    x_offset = width - logo_resized.shape[1] - 10
    y_offset = height - logo_resized.shape[0] - 10

    if logo_resized.shape[2] == 4:
        b, g, r, a = cv2.split(logo_resized)
        overlay_color = cv2.merge((b, g, r))
        alpha_mask = a
    else:
        overlay_color = logo_resized
        alpha_mask = np.ones((logo_resized.shape[0], logo_resized.shape[1]), dtype=np.uint8) * 255

    roi = frame[y_offset:y_offset + overlay_color.shape[0], x_offset:x_offset + overlay_color.shape[1]]

    if roi.shape[:2] != overlay_color.shape[:2]:
        overlay_color = cv2.resize(overlay_color, (roi.shape[1], roi.shape[0]))
        alpha_mask = cv2.resize(alpha_mask, (roi.shape[1], roi.shape[0]))

    alpha_mask = alpha_mask.astype(float) / 255
    for c in range(0, 3):
        roi[:, :, c] = (alpha_mask * overlay_color[:, :, c] + (1 - alpha_mask) * roi[:, :, c])

    frame[y_offset:y_offset + overlay_color.shape[0], x_offset:x_offset + overlay_color.shape[1]] = roi

    return frame


def recompile_video(frames_folder, output_video_path, fps=24):
    frames = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder)
                     if f.endswith('.jpg')], key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not frames:
        print("Error: No frames found in the folder!")
        return

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_video_path, codec="libx264")

# --- Process Video and Place Logo ---
def process_video_with_logo(video_path, logo_path, output_folder, scene_threshold=0.4, min_scene_length=30):
    cap = cv2.VideoCapture(video_path)
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

    if logo is None:
        raise FileNotFoundError(f"Logo file not found or could not be loaded: {logo_path}")

    scenes = detect_scenes(video_path, scene_threshold, min_scene_length)
    print(f"Detected {len(scenes)} scenes.")

    positions = ["top_left", "top_right", "bottom_left", "bottom_right"]
    frame_count = 0
    frame_files = []
    for scene_start, scene_end in scenes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, scene_start)
        for frame_idx in range(scene_start, scene_end + 1):
            ret, frame = cap.read()
            if not ret:
                break

            frame = overlay_logo_on_frame(frame, logo)

            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_files.append(frame_filename)
            frame_count += 1

    cap.release()
    print(f"Total frames processed: {frame_count}")

    recompile_video(output_folder, "output_video_with_logo.mp4", fps=24)

    # Clean up temporary folder
    shutil.rmtree(output_folder)
