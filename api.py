from flask import Flask, request, jsonify
from predict import pred_face
from model import check_db
from database import save_img_db
import base64
import logging

# pip install -r requirements.txt

app = Flask(__name__)

# defining logger file
logger = logging.getLogger(__name__)
# defines the the lowest-severity log message a logger will handle
logger.setLevel(logging.INFO)
# defines the format of our log messages
formatter = logging.Formatter('%(asctime)s :: %(module)s :: %(levelname)s :: %(message)s')
# define the file to which the logger will log
file_handler = logging.FileHandler('logger.log')
# setting up the format for the logger file
file_handler.setFormatter(formatter)
# adding FileHandler object to logger which would help us send logging output to disk file
logger.addHandler(file_handler)


def base64_to_image(base64_str):
    path_to_save = "imageToSave.jpg"
    with open(path_to_save, "wb") as fh:
        fh.write(base64.decodebytes(base64_str.encode('utf-8')))
    return path_to_save


@app.route('/', methods=['POST'])
def Hello():
    return jsonify("Welcome to Deep Learning Face Algorithm Created by Jatin Kaushik and Mohit Negi")


@app.route('/face_recogniser', methods=['POST'])  # for calling the API
def face_api():
    if (request.method == 'POST'):
        try:
            # getting json request data
            label = request.json['label']
            img_b64_str = request.json['img']
            # loging info
            logger.info("Face Logs :: The request has been received !!")
            img_path = base64_to_image(img_b64_str)
            db_table_name = 'Face_Embedding'
            result = pred_face(db_table_name, img_path, label)
            logger.info("Face Logs :: The output is :  {}".format(result))
            return jsonify({"User": result})
        except Exception as e:
            logger.info("Face Logs :: The output is :  {}".format(str(e)))
            return jsonify({"Status": "Error"})


@app.route('/add_face', methods=['POST'])  # for calling the API
def reg_user():
    if (request.method == 'POST'):
        # try:
            # getting json request data
            label = request.json['label']
            img_b64_str = request.json['img']
            img_path = base64_to_image(img_b64_str)
            db_table_name = 'Face_Embedding'
            message = save_img_db(img_path, label, db_table_name)
            # loging info
            if message is 'Success':
                logger.info("Face Logs :: 1 Image of {} Inserted Succesfully !!".format(label))
                result = "Success"
                logger.info("Face Logs :: The output is :  {}".format(result))
                return jsonify({"Status": result})
            else:
                logger.info("Face Logs :: No Face detected in Image of {}!!".format(label))
                result = "No Face Detected"
                return jsonify({"Status":result})
        # except Exception as e:
        #     logger.info("Face Logs :: The output is :  {}".format(str(e)))
        #     return jsonify({"Status": "Error"})


# @app.route('/add_array', methods=['POST'])  # for calling the API
# def add_array():
#     if (request.method == 'POST'):
#         try:
#             # getting json request data
#             label = request.json['label']
#             array_of_images = request.json['img']
#             db_table_name = 'Face_Embedding'
#             logger.info("{} Image's of {} Found in Request !!".format(len(array_of_images), label))
#             for b64_str in array_of_images:
#                 img_path = base64_to_image(b64_str)
#                 save_img_db(img_path, label, db_table_name)
#                 # loging info
#             logger.info("The request has been received !!")
#             result = "Success"
#             logger.info("The output is :  {}".format(result))
#             return jsonify({"Status": result})
#         except Exception as e:
#             logger.info("The output is :  {}".format(str(e)))
#             return jsonify({"Status": "Error"})


@app.route('/check_database', methods=['POST'])  # for calling the API
def check_DB():
    if (request.method == 'POST'):
        try:
            # loging info
            # getting json request data
            label = request.json['label']
            db_name = request.json['database']
            logger.info("{} :: Loading Database !!".format(db_name))
            db_table_name = db_name
            count_of_images = check_db(db_table_name, label)
            logger.info("{} :: The Total no. of Image's Found for {} is {}".format(db_name, label, count_of_images))
            return jsonify({"The Total no. of Image's Found for {} is ".format(label): count_of_images})
        except Exception as e:
            logger.info("{} :: The output is :  {}".format(db_name, str(e)))
            return jsonify({"Status": e})


if __name__ == '__main__':
    app.run(host="localhost", port=9000, debug=True)
