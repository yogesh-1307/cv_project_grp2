import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to upload image and apply transformations
@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No file uploaded.", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file.", 400
    
    # Save the uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Load the image
    image = cv2.imread(filepath)
    if image is None:
        return "Error loading the image. Ensure it is a valid image file.", 400
    
    # Get transformation choices
    transformations = request.form.getlist('transformations')
    rows, cols, _ = image.shape
    
    # Apply transformations
    processed_images = []
    
    if 'translate' in transformations:
        tx, ty = 50, 100  # Translate 50px right, 100px down
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
        processed_images.append(('Translated', translated_image))
    
    if 'rotate' in transformations:
        angle = 45  # Rotate 45 degrees
        center = (cols // 2, rows // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        processed_images.append(('Rotated', rotated_image))
    
    if 'scale' in transformations:
        scale_x, scale_y = 1.5, 1.5
        scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)
        processed_images.append(('Scaled', scaled_image))
    
    if 'shear' in transformations:
        shear_x, shear_y = 0.2, 0.3
        shearing_matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
        sheared_image = cv2.warpAffine(image, shearing_matrix, (cols, rows))
        processed_images.append(('Sheared', sheared_image))
    
    if 'flip' in transformations:
        flipped_image = cv2.flip(image, 1)
        processed_images.append(('Flipped', flipped_image))
    
    if 'crop' in transformations:
        cropped_image = image[50:200, 100:300]
        processed_images.append(('Cropped', cropped_image))
    
    if 'perspective' in transformations:
        pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
        pts2 = np.float32([[10, 100], [180, 50], [50, 250], [200, 220]])
        perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_image = cv2.warpPerspective(image, perspective_matrix, (cols, rows))
        processed_images.append(('Perspective', perspective_image))
    
    # Save processed images and generate HTML to display them
    results = []
    for name, img in processed_images:
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{name}.jpg")
        cv2.imwrite(processed_path, img)
        results.append((name, f"/processed/{name}.jpg"))
    
    return render_template('result.html', results=results)

# Route to serve processed images
@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
