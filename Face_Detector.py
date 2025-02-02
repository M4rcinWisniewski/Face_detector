import cv2
import numpy as np
import requests

# Function to load image from a URL using numpy
def load_image_from_url(url):
    response = requests.get(url)  # Download the image
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)  # Convert to NumPy array
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Decode the image
    return img

# URL of the image
image_url = input("Type in path to url to detect face: ") 

# Load image
img = load_image_from_url(image_url)

if img is None:
    print("Error: Image could not be loaded.")
    exit()

# Load pre-trained face detection model
face_data = cv2.CascadeClassifier("./assets/haarcascade_frontalface_default.xml")

# Convert to grayscale
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = face_data.detectMultiScale(grayscale)

# Draw rectangles around detected faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
    cv2.putText(img, "Face", (x, y + h + 30), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 2, cv2.LINE_AA)

# Display the image
cv2.namedWindow("Detected face in your image", cv2.WINDOW_NORMAL)  # Allows manual resizing
cv2.resizeWindow("Detected face in your image", 800, 600)  # Set max width and height
cv2.imshow("Detected face in your image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

