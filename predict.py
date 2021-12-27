from model import get_single_user_feature
from embeddings import get_embeddings
from extract_facesv2 import crop_faces
from scipy.spatial import distance


def pred_face(db_table_name, image_path, label):
    proba_score = []
    embedding_from_db = get_single_user_feature(db_table_name,label)
    for i in embedding_from_db:
        DB_vectors = i[1:]
        face_array = crop_faces(image_path)
        face_embedding = get_embeddings(face_array)
        # print(len(embedding_from_db),len(face_embedding))
        similarity = 1 - distance.cosine(DB_vectors, face_embedding)
        percentage = similarity * 100
        proba_score.append(percentage)
    #print(proba_score)
    #print("There are {} Image's found in Database Server of {}.".format(len(proba_score), label))
    # print(proba_score)
    if len(proba_score) > 0:
        if max(proba_score) > 70:
            return label
        else:
            return "Unknown Person"
    else:
        return "Unknown Person"

# print(pred_face('Face_Embedding', 'testjk.jpeg', 'jatin'))
