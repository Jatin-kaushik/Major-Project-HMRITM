import mysql.connector
from scipy.spatial import distance
from deepface import DeepFace

from embeddings import get_embeddings
from extract_facesv2 import crop_faces


def get_single_user_feature(db_table_name, label):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             port=3306,
                                             database='mydbtester',
                                             user='root',
                                             password='jatin281',
                                             auth_plugin='mysql_native_password')
        # print("connection succesfully created")
        connection.autocommit = True
        cursor = connection.cursor()
        sql = "SELECT * FROM `mydbtester`.`{}` WHERE Label = '{}'".format(db_table_name, label)
        cursor.execute(sql)
        x = cursor.fetchall()
        # print(x) #it Will return all data of specific Label from DB
        return x
    except Exception as e:
        return e
    finally:
        cursor.close()
        connection.close()


def check_db(db_table_name, label):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             port=3306,
                                             database='mydbtester',
                                             user='root',
                                             password='jatin281',
                                             auth_plugin='mysql_native_password')

        # print("connection succesfully created")
        connection.autocommit = True
        cursor = connection.cursor()
        sql = "SELECT * FROM `mydbtester`.`{}` WHERE Label = '{}'".format(db_table_name, label)
        cursor.execute(sql)
        x = cursor.fetchall()
        lenght = len(x)
        # print(x) #it Will return all data of specific Label from DB
        return lenght
    except Exception as e:
        return e
    finally:
        cursor.close()
        connection.close()


def verify_faces(image_path1, image_path2):
    face_array = crop_faces(image_path1)
    face_embedding1 = get_embeddings(face_array)
    face_array = crop_faces(image_path2)
    face_embedding2 = get_embeddings(face_array)
    # print(len(embedding_from_db),len(face_embedding))
    similarity = 1 - distance.cosine(face_embedding1, face_embedding2)
    # print(similarity)
    return similarity


def verify_deep(img1, img2):
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

    # face verification
    result = DeepFace.verify(img1_path=img1, img2_path=img2, model_name=models[1], distance_metric=metrics[1], detector_backend=backends[3])
    return result


def analyze_deep(img):
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
    obj = DeepFace.analyze(img_path=img, actions=['age', 'gender', 'race', 'emotion'], detector_backend=backends[3])
    return obj


# print(get_single_user_feature('Face_Embedding','jatin'))
# verify_faces('1.jpeg','2 (11).jpg')
# verify_faces('1.jpeg','2 (12).jpg')
# verify_faces('1.jpeg','2 (13).jpg')
# verify_faces('1.jpeg','2 (14).jpg')
# verify_faces('1.jpeg','2 (15).jpg')
# print(verify_deep('facedata/m1.jpg', 'facedata/r1.jpg'))
# print(analyze_deep('facedata/m1.jpg'))