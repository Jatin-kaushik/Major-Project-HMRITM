import mysql.connector
from embeddings import get_embeddings
from extract_facesv2 import crop_faces
import os


def create_table(table_name):
    connection = mysql.connector.connect(host='localhost',
                                         database='mydbtester',
                                         user='root',
                                         password='jatin281',
                                         auth_plugin='mysql_native_password')
    # print("connection succesfully created")
    connection.autocommit = True

    cursor = connection.cursor()
    cursor.execute("CREATE TABLE `mydbtester`.`{}` (`Label` VARCHAR(100) NOT NULL)".format(table_name))
    print("column created")
    for i in range(1, 129):
        tb = "ALTER TABLE `mydbtester`.`{}` ADD COLUMN `Embedded_id".format(table_name) + str(i) + "`" + " FLOAT NOT NULL"
        cursor.execute(tb)
    print(tb)
    cursor.close()
    connection.close()
    print('databases')


def save_img_db(image_path, label, tablename):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='mydbtester',
                                             user='root',
                                             password='jatin281',
                                             auth_plugin='mysql_native_password')
        # print("connection succesfully created")
        connection.autocommit = True
        face_array = crop_faces(image_path)
        cursor = connection.cursor()
        if face_array is not None:
            face_embedding = get_embeddings(face_array)
            # print(face_embedding.shape)

            insert_statement = "INSERT INTO `mydbtester`.`{}` VALUES ( '%s', ".format(tablename) % label
            for i in face_embedding:
                insert_statement += str(i) + ", "
            insert_statement = insert_statement[:-2]
            insert_statement = insert_statement + ')'
            # print(insert_statement)
            #print("Row inserted")  # INSERT INTO `mydbtester`.`emb_db1` (`Embedded_id1`,`Embedded_id2`) VALUES (1,2);
            cursor.execute(insert_statement)
            return "Success"
        else:
            print("No Face Detected in Photo {}".format(label))
    except Exception as e:
        return e


def load_directory(folder, tablename):
    people = os.listdir(folder)
    print("{} people found: ".format(len(people)), people)
    for subdir in people:
        # path
        _path = os.path.join(folder, subdir)

        # skip any files that are not dir
        if not os.path.isdir(_path):
            # skip
            continue
        image_paths = os.listdir(_path)
        print("{} Images Found For {}.".format(len(image_paths), subdir))
        for filename in image_paths:
            # path of image
            path_image = os.path.join(_path, filename)
            save_img_db(path_image, subdir, tablename)

# create table using tablename
# create_table('Face_Embedding')
# load_directory(r'C:\Users\Jatin kaushik\Downloads\Dataset\train', 'Face_Embedding')

# save_img_db('test_face.jpg', 'Mohit','Face_Embedding_Anywhere')