from flask import Flask, request, jsonify, render_template
import uuid
import os
import json
from google.cloud import pubsub_v1
import google.auth
from google.auth.transport.requests import Request
from google.auth import exceptions
from werkzeug.utils import secure_filename
import logging
from config import Config
import psycopg2
from psycopg2.extras import DictCursor
app = Flask(__name__)

# Set maximum content length for file uploads (e.g., 50MB max)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Define the folder to store the uploaded files temporarily
UPLOAD_FOLDER = 'data/real_measurement'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Google Cloud Pub/Sub settings
PROJECT_ID = Config.PROJECT_ID
PUBSUB_TOPIC = Config.PUBSUB_TOPIC

# Authenticate using the service account JSON key (load credentials from a file)
def authenticate_google_cloud():
    try:
        # The `google.auth` library will automatically use the GOOGLE_APPLICATION_CREDENTIALS environment variable
        # for authentication. If not, provide the service account key directly.
        credentials, project = google.auth.load_credentials_from_file("/home/sapa6220/ad-placement/data/ad-placement-443705-a5005c45a272.json")

        # Initialize the Pub/Sub publisher client using the credentials
        publisher = pubsub_v1.PublisherClient(credentials=credentials)
        topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)
        
        return publisher, topic_path
    except exceptions.DefaultCredentialsError as e:
        logging.error(f"Authentication failed: {e}")
        return None, None

# Allowed file extensions for video and logo
ALLOWED_EXTENSIONS = {'mp4', 'png'}

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html")

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            connect_timeout=30
        )
        return conn
    except psycopg2.Error as e:
        print("Unable to connect to the database:", e)
        return None

@app.route('/videos')
def videos():
    connection = get_db_connection()
    if connection is None:
        return "Error connecting to the database.", 500  # Return a server error

    try:
        with connection.cursor(cursor_factory=DictCursor) as cursor:
            sql = "SELECT * FROM videos"
            cursor.execute(sql)
            videos = cursor.fetchall()
        return render_template('index2.html', videos=videos)
    finally:
        connection.close()  # Ensure the connection is closed

@app.route('/process-video', methods=['POST'])
def process_video():
    try:
        logging.info("Processing video request...")

        # Get video and logo files and their descriptions from the request
        video_file = request.files.get('video')
        logo_file = request.files.get('logo')
        video_description = request.form.get('video_description')
        logo_description = request.form.get('logo_description')

        if not video_file or not logo_file or not video_description or not logo_description:
            logging.error("Error: Missing required fields")
            return jsonify({"error": "Missing required fields (video, logo, video description, or logo description)"}), 400

        logging.info(f"Received video file: {video_file.filename}, logo file: {logo_file.filename}")
        logging.info(f"Video Description: {video_description}, Logo Description: {logo_description}")

        # Generate unique video ID
        video_id = str(uuid.uuid4())
        logging.info(f"Generated video ID: {video_id}")

        # Define the file paths to store the files temporarily
        video_filename = secure_filename(video_file.filename)
        logo_filename = secure_filename(logo_file.filename)
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{video_filename}")
        logo_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{logo_filename}")

        # Save the video and logo files temporarily
        video_file.save(video_path)
        logo_file.save(logo_path)

        logging.info(f"Video saved to {video_path}")
        logging.info(f"Logo saved to {logo_path}")

        # Authenticate with Google Cloud and get the Pub/Sub publisher
        publisher, topic_path = authenticate_google_cloud()
        if publisher is None or topic_path is None:
            return jsonify({"error": "Google Cloud authentication failed"}), 500

        # Send message to Pub/Sub queue to trigger video processing
        message = {
            "video_id": video_id,
            "video_path": video_path,
            "logo_path": logo_path,
            "video_description": video_description,
            "logo_description": logo_description
        }
        message_data = json.dumps(message).encode("utf-8")  # Convert the message to bytes
        publisher.publish(topic_path, data=message_data)  # Send to Pub/Sub

        logging.info(f"Message sent to Pub/Sub with video_id: {video_id}")

        return jsonify({"message": "Video processing started", "video_id": video_id}), 202

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": "An error occurred during video processing"}), 500


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
