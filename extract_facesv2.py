# function for face detection with mtcnn
from PIL import Image
import numpy as np
import cv2

from mtcnn.mtcnn import MTCNN

# create the detector, using default weights
detector = MTCNN()


def detect_face(image_array):
    # detect faces in the image
    results = detector.detect_faces(image_array)
    if len(results) > 0:

        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']

        # convert the co-ordinates into cropping format
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # crop and extract the face
        face = image_array[y1:y2, x1:x2]

        return face
    else:
        return None


# extract a single face from a given photograph
def crop_faces(file, required_size=(160, 160)):
    # load image from file
    image = Image.open(file)

    # convert to RGB, if needed
    image = image.convert('RGB')

    # convert to array
    image_array = np.asarray(image)

    # face extracted from MTCNN
    cropped_face = detect_face(image_array)

    if cropped_face is not None:
        # resize array to the model size by reading the array into Pil image object
        image = Image.fromarray(cropped_face)
        image = image.resize(required_size)

        face_array = np.asarray(image)
        return face_array
    else:
        return None


def save_face(array, name="Cropped Face.jpg"):
    cv2.imwrite(name, cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def plot_face(array):
    pass

# load the photo and extract the face
# pixels = crop_faces('C:\\Users\\Jatin kaushik\\Downloads\\Dataset\\SIT.png')
# print(pixels)
# save_face(pixels)
# plot_face(pixels)
