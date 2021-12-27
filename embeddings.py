from keras.models import load_model
import numpy as np
import os

path = os.path.join('models', 'facenet_keras.h5')
#print(path)
# loading the facenet model
emb_model = load_model(path)

# summarize input and output shape
print(emb_model.inputs)
print(emb_model.outputs)


def get_embeddings(face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)

    y_emb = emb_model.predict(samples)

    # get embedding
    embedding = y_emb[0]
    return embedding
