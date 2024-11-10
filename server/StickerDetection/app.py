# Import the required libraries
import numpy as np
import cv2
from ultralytics import YOLO

# Import firebase modules
import firebase_admin
from firebase_admin import credentials
import pyrebase


# Initialize firebase app with credentials
cred = credentials.Certificate(
    '')  # change this to your own service account key file
firebase_admin.initialize_app(cred, {
    'storageBucket': ''  # change this to your own project id
})

# Initialize pyrebase app with configuration
config = {
    "apiKey": "",  # change this to your own api key
    "authDomain": "",  # change this to your own project id
    "databaseURL": "",
    "storageBucket": ""  # change this to your own project id
}

pyrebase_app = pyrebase.initialize_app(config)

# Get the storage reference from pyrebase app
storage = pyrebase_app.storage()
model = YOLO("yolov5s.pt")

# Load the custom model
custom_model = YOLO("") # add your custom yolo model path
custom_model.conf = 0.98  # or any value between 0 and 1
custom_model.max_det = 1
custom_model.iou = 0.7


# Define a function to get the center point of a bounding box
def get_center(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy


# Pass the car region in the custom model to detect object in car region
def detect_object_in_car_region(img, box):
    # Crop the car region from the image
    x1, y1, x2, y2 = box
    x1 = int(x1 + 10)
    y1 = int(y1 + 10)
    x2 = int(x2 + 10)
    y2 = int(y2 + 10)
    car_region = img[y1 + 10: y2 - 10, x1 + 10: x2 - 10]

    # Run inference on the car region and get the results
    results = custom_model(car_region, device="mps", imgsz=640, conf=0.78, max_det=1)
    # Return the results
    return results
# Define a global variable for the car counter
car_counter = 0
# Define a function to assign a unique id to each car based on the center point
def assign_id(centers):
    global prev_centers, prev_ids  # Use global variables to store the previous centers and ids
    ids = []
    for i, c1 in enumerate(centers):
        # If the center is close to any previous center, assign the same id
        for j, c2 in enumerate(prev_centers):
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            if dist < 15:  # You can adjust this threshold as needed
                ids.append(prev_ids[j])
                break
        else:
            # Otherwise, assign a new id that is not used before
            new_id = len(prev_ids) + 1
            while new_id in prev_ids:
                new_id += 1
            ids.append(new_id)
        # If the center is close to any other center in the current frame, assign the same id
        for k, c3 in enumerate(centers[:i]):
            dist = np.linalg.norm(np.array(c1) - np.array(c3))
            if dist < 15:  # You can adjust this threshold as needed
                ids[i] = ids[k]
                break
    # Update the previous centers and ids with the current ones
    prev_centers = centers.copy()
    prev_ids = ids.copy()
    return ids


# Draw the car counter on the frame
def draw_counter(img):
    global car_counter  # Use the global variable for the car counter
    # Put the car counter text on the top left corner of the frame
    cv2.putText(img, f'Car Counter: {car_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# Define a global variable for the car counter
car_counter = 0

# Define a line to count cars when they cross it
line_x1 = 1100
line_x2 = 1500
line_y = 900

# Initialize the global variable for previous boxes
prev_boxes = {}
# Define a global variable for the saved frames
saved_frames = []
keys = {}
saved_ids = []

# Import the datetime library
import datetime

area_threshold = 600000  # or any value you like
area_threshold_2 = 150000  # or any value you like

# Import the datetime library
import datetime


# Draw the box and center point for each object detected in car region
def draw_object_in_car_region(img, box, results, id, label_):
    my_dict = {2: "car", 5: "bus", 7: "truck"}
    # Create a scalar tensor

    # Convert it to a Python number
    n = label_.item()
    # Crop the car region from the image
    cy = 0
    cx = 0
    # Crop the car region from the image
    x1, y1, x2, y2 = box
    x1 = int(x1 + 10)
    y1 = int(y1 + 10)
    x2 = int(x2 + 10)
    y2 = int(y2 + 10)
    car_region = img[y1 + 10: y2 - 10, x1 + 10: x2 - 10]

    # Get the bounding boxes and labels of the detected objects in car region
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    labels = results[0].boxes.cls

    # Draw the box and center point for each object detected in car region
    for box, label in zip(boxes, labels):
        name = results[0].names[int(label)]

        # Draw a rectangle around the box
        cv2.rectangle(car_region, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        # Get the center point of the box
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        # Draw a circle at the center point
        cv2.circle(car_region, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        # Write names and center points
        cv2.putText(car_region, f"{name}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)


    key_ = None  # Get the database reference from pyrebase app
    db = pyrebase_app.database()
    x1, y1, x2, y2 = box
    x1 = int(x1 + 10)
    y1 = int(y1 + 10)
    x2 = int(x2 + 10)
    y2 = int(y2 + 10)

    if id not in saved_frames:
        # Get the current time as a string
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Save the car region with the id and the time as the file name
        #  cv2.imwrite(f'{id}_{now}.jpg', car_region)
        # Get the storage reference from pyrebase app
        storage = pyrebase_app.storage()

        # Upload the image file to firebase storage with a unique name based on the vehicle id
        storage.child(f'/images/vehicles/{id}_{now}.jpg').put(f'{id}_{now}.jpg')

        # Get the download url of the image file from firebase storage
        url = storage.child(f'/images/vehicles/{id}_{now}.jpg').get_url(None)

        # Create a dictionary with the url and time as values
        data = {"url": url, "time": time, "detected": False, "vehicleType": my_dict[n]}
        # (data)

        # Push the data to the notification child in the realtime firebase database
        ref = db.child("notification").push(data)

        # Get the key of the data from the dictionary
        key = ref['name']
        keys[id.item()] = key

        #    id_ = id.item()

        # Add the id to the saved frames
        saved_frames.append(id_)


# Load a video of vehicles
cap = cv2.VideoCapture('')

# Initialize the global variables for previous centers and ids
prev_centers = []
prev_ids = []
tracker = "bytetrack.yaml"
prev_vehicles = {}
id_outside = None
counter = 0
frame_count = {}  # new
values = []
counter = 0
db = pyrebase_app.database()
my_dict = {2: "car", 5: "bus", 7: "truck"}

# Loop over the frames of the video
while cap.isOpened():
    # Read a frame from the video
    ret, img = cap.read()
    if not ret:
        break
    res = model.track(img, conf=0.90, persist=True, tracker=tracker, max_det=1, classes=[2, 5, 7])
    obj_id = res[0].boxes.data[:, 4]
    boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
    labels = res[0].boxes.cls
    # Filter out the boxes and labels that are not cars
    car_boxes = []
    # Check if no objects detected in the frame and set saved_frames to be empty
    ids = []
    car_labels = []
    for box, label, id in zip(boxes, labels, obj_id):
        car_boxes.append(box)
        car_labels.append(label)
        ids.append(id)
    # Get the center points of the car boxes
    car_centers = [get_center(box) for box in car_boxes]

    # Assign a unique id to each car
    car_ids = assign_id(car_centers)

    # Call the save_frame function for each car detected with the box and id
    for box, label, center, id in zip(car_boxes, car_labels, car_centers, ids):
        x1, y1, x2, y2 = box
        x1 = int(x1 + 10)
        y1 = int(y1 + 10)
        x2 = int(x2 + 10)
        y2 = int(y2 + 10)
        id_outside = id
        n = label.item()

        area = (x2 - x1) * (y2 - y1)
        if area > area_threshold_2:
            if id not in saved_ids:
                # If not, crop and save the image as before
                car_region = img[y1:y2, x1:x2]
                cv2.imwrite(f'{id}_new.jpg', car_region)

                # Add the current id to the saved_ids set
                saved_ids.append(id)

        if area > area_threshold:
            car_region = img[y1:y2, x1:x2]

            # Detect object in car region
            results = detect_object_in_car_region(img, box)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            # Draw the box and center point for each object detected in car region
            for box, label in zip(boxes, labels):
                # Draw a rectangle around the box
                cv2.rectangle(car_region, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                # Get the center point of the box
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                # Draw a circle at the center point
                cv2.circle(car_region, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                # Write names and center points
            # Check if the id is new or not
            if id not in prev_vehicles:
                # Add it to the prev_vehicles dictionary with a boolean value indicating if any object is detected or not
                prev_vehicles[id] = len(boxes) > 0
                counter = 0
                # Initialize the number of frames for this vehicle id to zero
                frame_count[id] = 0  # new
            else:
                # If not new, get the previous boolean value from the prev_vehicles dictionary
                prev_detected = prev_vehicles[id]

                # Check if any object is detected or not
                if len(boxes) > 0:
                    # Update the prev_vehicles dictionary with a new boolean value indicating that an object is detected
                    prev_vehicles[id] = True

                else:
                    pass

    counter = 0
    values = []
    prev_id = None  # Initialize a variable to store the previous id
    for id in prev_vehicles:
        # Compare the current id with the previous id
        if id != prev_id:
            # If they are different, reset the counter and the values list
            counter = 0
            values = []
        counter = counter + 1
        values.append(prev_vehicles.get(id))

        if counter == 10:
            result = all(value == False for value in values)
            if result:
                if id not in saved_frames:
                    # Get the storage reference from pyrebase app
                    storage = pyrebase_app.storage()

                    # Upload the image file to firebase storage with a unique name based on the vehicle id
                    storage.child(f'/images/vehicles/{id}_new.jpg').put(f'{id}_new.jpg')

                    # Get the download url of the image file from firebase storage
                    url = storage.child(f'/images/vehicles/{id}_new.jpg').get_url(None)

                    # Get the current time as a string
                    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Create a dictionary with the url and time as values
                    data = {"url": url, "time": time, "detected": False, "vehicleType": my_dict[n]}
                    # Push the data to the notification child in the realtime firebase database
                    ref = db.child("notification").push(data)

                    # Save the car region with the id and the time as the file name
                    cv2.imwrite(f'{id}_new.jpg', img)

                    id_ = id.item()

                    # Add the id to the saved frames
                    saved_frames.append(id_)

            else:
                if id not in saved_frames:

                    # Get the storage reference from pyrebase app
                    storage = pyrebase_app.storage()

                    #     # Upload the image file to firebase storage with a unique name based on the vehicle id
                    storage.child(f'/images/vehicles/{id}_new.jpg').put(f'{id}_new.jpg')

                    # Get the download url of the image file from firebase storage
                    url = storage.child(f'/images/vehicles/{id}_new.jpg').get_url(None)

                    # Get the current time as a string
                    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Create a dictionary with the url and time as values
                    data = {"url": url, "time": time, "detected": True, "vehicleType": my_dict[n]}
                    # (data)

                    # Push the data to the notification child in the realtime firebase database
                    # Push the data to the notification child in the realtime firebase database
                    ref = db.child("notification").push(data)

                    # Save the car region with the id and the time as the file name
                    cv2.imwrite(f'{id}_new.jpg', img)

                    id_ = id.item()

                    # Add the id to the saved frames
                    saved_frames.append(id_)
        # Update the previous id with the current id
        prev_id = id

    # Draw the boxes, labels and ids on the frame
    for box, label, center, id in zip(car_boxes, car_labels, car_centers, ids):
        # Draw a rectangle around the box
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        # Draw a circle at the center point
        cv2.circle(img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        # Put the label and id text above the box
        cv2.putText(img, f'Car ', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1)

    # Show the frame with the detections
    cv2.imshow('Vehicle Detection', img)
    # Call the draw_counter function after drawing the boxes, labels and ids on the frame
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
