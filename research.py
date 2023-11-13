import dlib
import cv2
from scipy.spatial import distance

# Load the image
image = cv2.imread('/Users/apple/Desktop/Research/img1.jpeg')

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/apple/Desktop/Research/shape_predictor_68_face_landmarks.dat')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Ensure at least one face was found
if len(faces) > 0:
    # Assuming there's only one face detected in the image
    face = faces[0]

    # Get the landmarks for the detected face
    landmarks = predictor(gray, face)

    # Get the coordinates of point 25 and point 47
    x25, y25 = landmarks.part(25).x, landmarks.part(25).y
    x47, y47 = landmarks.part(47).x, landmarks.part(47).y

    # Calculate the Euclidean distance between point 25 and point 47
    distance_25_to_47 = distance.euclidean((x25, y25), (x47, y47))

    # Print and store the distance
    print(f"Distance between point 25 and point 47: {distance_25_to_47}")

    # Draw landmarks on the image
    for i in range(68):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

    # Display the image with facial landmarks
    cv2.imshow("Image with Facial Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face found in the image")
