import json
import os
import cv2
import numpy as np
from google.cloud import storage
from google.cloud import pubsub_v1
import google.auth
from moviepy.editor import *
from openai import OpenAI
from config import Config
import logging
import psycopg2

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = Config.PROJECT_ID
PUBSUB_TOPIC = Config.PUBSUB_TOPIC
BUCKET_NAME = Config.BUCKET_NAME


# Initialize OpenAI client
client = OpenAI(api_key=Config.API_KEY)

# Authenticate using service account
def authenticate_google_cloud():
    try:
        credentials, project = google.auth.load_credentials_from_file("/home/sapa6220/ad-placement/data/ad-placement-443705-a5005c45a272.json")
        storage_client = storage.Client(credentials=credentials, project=project)
        subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
        logger.info("Successfully authenticated Google Cloud services.")
        return storage_client, subscriber
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return None, None

storage_client, subscriber = authenticate_google_cloud()

# Function to generate advertisement
def generate_advertisement(video_description,logo_description):
    """
    Generate a catchy advertisement phrase based on the provided video description and product category/logo.
    """
    work_exp_messages = [
        {"role": "system", "content": "You are a helpful assistant that creates natural, non-intrusive advertisements."},
        {"role": "user", "content": f"Create a catchy advertisement phrase of 5 words or less for a product or service  using the brand/logo '{logo_description}', based on the following video description: '{video_description}'. The phrase should be short, impactful, and align naturally with the video content."}
    ]
    
    try:
        logger.info("Generating advertisement using OpenAI...")
        response_work_exp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=work_exp_messages,
            max_tokens=50,
            temperature=0.7
        )
        
        ad_text = response_work_exp.choices[0].message.content.strip()
        logger.info(f"Generated advertisement phrase: {ad_text}")
        return ad_text
    except Exception as e:
        logger.error(f"Error generating advertisement: {e}")
        return "Sorry, I couldn't generate the ad at this time."

# Function to detect activity in video frames
def detect_activity_in_frame(frame):
    height, width = frame.shape[:2]
    top_half = frame[:height // 2, :]
    bottom_half = frame[height // 2:, :]

    gray_top = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
    gray_bottom = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)

    top_activity = np.sum(np.abs(gray_top))
    bottom_activity = np.sum(np.abs(gray_bottom))

    return top_activity, bottom_activity

# Function to overlay logo on frame
def overlay_logo_on_frame(frame, logo, position="bottom_right"):
    height, width = frame.shape[:2]
    logo_resized = cv2.resize(logo, (int(width * 0.2), int(logo.shape[0] * (width * 0.2) / logo.shape[1])))

    if position == "bottom_right":
        x_offset = width - logo_resized.shape[1] - 10
        y_offset = height - logo_resized.shape[0] - 10
    else:
        x_offset = 10
        y_offset = height - logo_resized.shape[0] - 10

    if logo_resized.shape[2] == 4:
        b, g, r, a = cv2.split(logo_resized)
        overlay_color = cv2.merge((b, g, r))
        alpha_mask = a
    else:
        overlay_color = logo_resized
        alpha_mask = np.ones((logo_resized.shape[0], logo_resized.shape[1]), dtype=np.uint8) * 255

    roi = frame[y_offset:y_offset + overlay_color.shape[0], x_offset:x_offset + overlay_color.shape[1]]
    alpha_mask = alpha_mask.astype(float) / 255

    for c in range(0, 3):
        roi[:, :, c] = (alpha_mask * overlay_color[:, :, c] + (1 - alpha_mask) * roi[:, :, c])

    frame[y_offset:y_offset + overlay_color.shape[0], x_offset:x_offset + overlay_color.shape[1]] = roi
    return frame

# Function to overlay ad phrase on frame
def overlay_ad_phrase_on_frame(frame, ad_phrase):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2

    text_size = cv2.getTextSize(ad_phrase, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = 50

    cv2.putText(frame, ad_phrase, (text_x, text_y), font, font_scale, color, thickness)
    return frame

# Function to update the video metadata in Cloud SQL
def add_metadata_in_db(video_id, processed_video_url, video_description, logo_description):
    try:
        logger.info("Attempting to connect to the database...")
        
        # Connect to Cloud SQL (PostgreSQL)
        conn = psycopg2.connect(
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            connect_timeout = 30
        )
        
        logger.info("Successfully connected to the database.")
        
        cursor = conn.cursor()

        # Prepare the SQL insert query
        insert_query = """
        INSERT INTO videos (video_id, video_description, logo_description, processed_video_path)
        VALUES (%s, %s, %s, %s);
        """
        
        # Log before executing the insert query
        logger.info(f"Executing insert query for video ID: {video_id}")
        
        cursor.execute(insert_query, (video_id, video_description, logo_description, processed_video_url))

        # Commit the transaction - 
        conn.commit()

        # Log successful insertion
        logger.info(f"Video with ID {video_id} has been successfully updated in the database.")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        # Log any error that occurs
        logger.error(f"Error occurred while updating video in the database: {e}")
        logger.debug("Detailed error information", exc_info=True)

# Process the video with logo and ad phrase
def process_video_with_dynamic_logo_and_ad(video_path, logo_path, output_folder, video_description, logo_description):
    try:
        logger.info(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

        if logo is None:
            logger.error(f"Logo file not found or could not be loaded: {logo_path}")
            raise FileNotFoundError(f"Logo file not found or could not be loaded: {logo_path}")

        ad_phrase = generate_advertisement(video_description, logo_description)
        frame_count = 0
        frame_files = []

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info(f"End of video reached, total frames processed: {frame_count}")
                break

            top_activity, bottom_activity = detect_activity_in_frame(frame)
            position = "top_center" if top_activity < bottom_activity else "bottom_center"
            frame = overlay_logo_on_frame(frame, logo, position=position)
            frame = overlay_ad_phrase_on_frame(frame, ad_phrase)

            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            final_video_save_path = os.path.join(output_folder,"output_video_with_dynamic_logo_and_ad.mp4")
            cv2.imwrite(frame_filename, frame)
            frame_files.append(frame_filename)
            frame_count += 1

        cap.release()
        recompile_video(output_folder, final_video_save_path, fps=24)

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        raise e

# Recompile the frames into a video
def recompile_video(frames_folder, output_video_path, fps=24):
    try:
        logger.info("Recompiling video from frames...")
        frames = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.jpg')],
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))

        if not frames:
            logger.error("Error: No frames found in the folder!")
            return

        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(output_video_path, codec="libx264")
        logger.info(f"Video compiled successfully: {output_video_path}")
    except Exception as e:
        logger.error(f"Error recompiling video: {e}")
        raise e

# Worker function to process video messages from Pub/Sub
def process_video(message):
    try:
        logger.info("Received message from Pub/Sub.")
        message_data = json.loads(message.data.decode('utf-8'))

        video_id = message_data['video_id']
        video_path = message_data['video_path']
        logo_path = message_data['logo_path']
        video_description = message_data.get('video_description', '')
        logo_description = message_data.get('logo_description', '')

        output_folder = f"data/processed_{video_id}"
        os.makedirs(output_folder, exist_ok=True)

        logger.info(f"Processing video for video_id: {video_id}")
        process_video_with_dynamic_logo_and_ad(video_path, logo_path, output_folder, video_description, logo_description)

        # Upload the processed video to Google Cloud Storage
        bucket = storage_client.bucket(BUCKET_NAME)
        processed_blob = bucket.blob(f"processed_videos/{video_id}_processed_video.mp4")
        processed_blob.upload_from_filename(f"{output_folder}/output_video_with_dynamic_logo_and_ad.mp4")

        processed_video_url = processed_blob.public_url
        logger.info(f"Processed video uploaded to: {processed_video_url}")

        # Update video details in Cloud SQL
        add_metadata_in_db(video_id, processed_video_url, video_description, logo_description)

        message.ack()

    except Exception as e:
        logger.error(f"Error processing video for video_id {video_id}: {e}")
        message.nack()

if __name__ == '__main__':
    subscription_path = subscriber.subscription_path(PROJECT_ID, PUBSUB_TOPIC)
    future = subscriber.subscribe(subscription_path, callback=process_video)
    logger.info(f"Worker started, listening for messages on {subscription_path}.")
    
    try:
        future.result()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")
    except Exception as e:
        logger.error(f"Error in worker: {e}")
