# Non-Intrusive Ad Generation

By :
   ### Santhosh Pattamudu Manoharan
   ### Harini Sai Padamata

## Goal
Transform any short vertical video into an advertisement.

## Steps to Run the Code

1. **Create a Python Virtual Environment:**

2. **Install Requirements:**
   ```
   pip install -r requirements.txt
   ```

3. **Set Up Services in Google Cloud Platform (GCP) and OpenAI :**
   - Configure Pub/Sub.
   - Set up Cloud SQL.
   - Set up Cloud Storage.
   - Create an API Key from OpenAI

4. **Configure the Application:**
   - Enter the necessary details in the `config.py` file.

5. **Run the Flask Server:**
   ```
   python main.py
   ```

6. **Run the Worker Service:**
   ```
   python worker.py
   ```
   The worker listens to the Flask server and processes the video.

7. **Access the Application:**
   - Navigate to `http://localhost:5000` in your browser.

8. **Upload Video and Logo:**
   - Provide the video, video description, logo, and logo description.

9. **View Processed Outputs:**
   - Visit the `/videos` endpoint to see all processed videos.
  
  
