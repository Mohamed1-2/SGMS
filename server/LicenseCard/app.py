import numpy as np
import cv2
from flask import Flask, jsonify, request
from tensorflow.lite.python.interpreter import Interpreter
from werkzeug.utils import secure_filename
import os
import pytesseract
import re

app = Flask(__name__)

# Define the directory to save the uploaded images
UPLOAD_FOLDER = 'flask/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Define the allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}




# Define a helper function to check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
number_regex = re.compile(r'\d+')

# Define the object detection function
model_path = "/flask/detect.tflite"
label_path = "/flask/classes.txt"
# Load the label map into memory
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


def detect_objects(image_path):
     # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No image file included in the request'}), 400

    # Get the file from the request
    file = request.files['file']

    # Check if the file has a valid filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save the file to the server
    filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filename)

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Find the ID of the logo class in the label map
    logo_class_name = "logo"
    logo_class_id = None
    for i, label in enumerate(labels):
        if label == logo_class_name:
            logo_class_id = i
            break

    highest_scores = {}
    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

    detections = []

    # Set minimum confidence thresholds
    min_conf_all = 0.3  # Minimum confidence threshold for all classes
    min_conf_logo = 0.9  # Minimum confidence threshold for logo class

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        # Check if the detected object is a logo
        if classes[i] == logo_class_id and scores[i] > min_conf_logo:
            # Get object name and bounding box coordinates
            object_name = labels[int(classes[i])]
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # Check if object already exists in dictionary
            if object_name in highest_scores:
                # Update score if new score is higher
                if scores[i] > highest_scores[object_name][0]:
                    highest_scores[object_name] = [scores[i], xmin, ymin, xmax, ymax]
            else:
                highest_scores[object_name] = [scores[i], xmin, ymin, xmax, ymax]

        # Check for other classes
        elif scores[i] > min_conf_all:
            # Get object name and bounding box coordinates
            object_name = labels[int(classes[i])]
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # Check if object already exists in dictionary
            if object_name in highest_scores:
                # Update score if new score is higher
                if scores[i] > highest_scores[object_name][0]:
                    highest_scores[object_name] = [scores[i], xmin, ymin, xmax, ymax]
            else:
                highest_scores[object_name] = [scores[i], xmin, ymin, xmax, ymax]

        else:
            break

    # Draw the boxes and labels for objects with the highest scores
    if len(highest_scores) == 4:
        for object_name, values in highest_scores.items():
            score, xmin, ymin, xmax, ymax = values
            #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            detections.append([object_name, score, xmin, ymin, xmax, ymax])

    # Create a list to store the extracted text with object name
    text_list = []
    for detection in detections:
        object_name = detection[0]
        score = detection[1]
        xmin = detection[2]
        ymin = detection[3]
        xmax = detection[4]
        ymax = detection[5]
        #  print(object_name)
        pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
        # Check if object is not a logo and is of 'id_number' class
        if object_name != 'logo' and object_name == 'id_number':
            # Crop image within bounding box coordinates
            object_img = image[ymin:ymax, xmin:xmax]
            #  cv2.imshow(object_img)

            gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

            # Apply OCR to extract text from cropped image
            text = pytesseract.image_to_string(gray)

            # Extract only numbers from the extracted text
            numbers = number_regex.findall(text)
            # Append object name and extracted text to text list
            text_list.append(f"{object_name}: {numbers}")
            # Print object name and extracted numbers
        # Check if object is not a logo and is of 'address' class
        elif object_name != 'logo' and object_name == 'address':
            # Crop image within bounding box coordinates
            object_img = image[ymin:ymax, xmin:xmax]
            gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

            # Apply OCR to extract text from cropped image
            text = pytesseract.image_to_string(gray)

            # Remove "Alamat / Address" from the extracted text
            text = text.replace("Alamat / Address", "")

            # Append object name and extracted text to text list
            text_list.append(f"{object_name}: {text}")
        elif object_name != 'logo':
            # Crop image within bounding box coordinates
            object_img = image[ymin:ymax, xmin:xmax]
            gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

            # Apply OCR to extract text from cropped image
            text = pytesseract.image_to_string(gray)
            # Append object name and extracted text to text list
            text_list.append(f"{object_name}: {text}")

    return text_list


# Define the API endpoint
@app.route('/detect_objects', methods=['POST'])
def detect_objects_api():
    # Receive the image from the Flutter app
    file = request.files['file']
    # Save the file to the server
    filename = secure_filename(file.filename)
    print(filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Detect objects in the image
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as f:

        image_data = f.read()
        text_list  = detect_objects(image_data)

    # Send back the results
    response = {'text_list': text_list}
    return jsonify(response)


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0')

