import requests
from flask import Flask, request, render_template, send_file
import numpy as np
import cv2
import os
import tempfile
import base64

# Initialize Flask app
app = Flask(__name__)

# Paths to model files
DIR = r"D:\image_colorization"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Load the colorization model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Home route to render the upload page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload or URL input and colorization
@app.route('/colorize', methods=['POST'])
def colorize_image():
    temp_path = None

    # Handle file upload
    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

    # Handle image URL
    elif 'image_url' in request.form and request.form['image_url']:
        image_url = request.form['image_url']
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
        else:
            return "Unable to download image from URL", 400

    else:
        return "No file uploaded or URL provided", 400

    # Process the uploaded or downloaded image
    image = cv2.imread(temp_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Convert image to base64 to display in HTML
    _, buffer = cv2.imencode('.jpg', colorized)
    colorized_base64 = base64.b64encode(buffer).decode('utf-8')

    # Convert the original image to base64
    _, buffer = cv2.imencode('.jpg', image)
    original_base64 = base64.b64encode(buffer).decode('utf-8')

    # Clean up temporary file
    os.remove(temp_path)

    # Render the template with images
    return render_template('index.html', original_image=original_base64, colorized_image=colorized_base64)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
