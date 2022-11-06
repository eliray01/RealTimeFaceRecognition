import cv2
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize,l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model

import pickle

import time

from mask_rcnn_images import procces




confidence_t=0.99
recognition_t=0.5
required_size = (160,160)


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img ,detector,encoder,encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    a_coords = []
    b_coords = []
    names = []
    a = (0,0)
    b = (0,0)
    #name = 'unknown'
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            a = pt_1
            b = pt_2
        else:
            cv2.rectangle(img, pt_1, pt_2, (0 ,255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
            a = pt_1
            a_coords.append(a)
            b = pt_2
            b_coords.append(b)
            names.append(name)
    return img, a_coords,b_coords, names

if __name__ == "__main__":

    #FACE RECOGNITION IMPORTS
    required_shape = (160,160)
    face_encoder = InceptionResNetV2()
    path_m = "weights/facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)

    #INSTANCE SEGMENTATION IMPORTS
    path_to_frozen_inference_graph = 'weights/frozen_inference_graph_coco.pb'
    path_coco_model = 'weights/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

    net = cv2.dnn.readNetFromTensorflow(path_to_frozen_inference_graph, path_coco_model)

    # # # If input video
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # output_movie = cv2.VideoWriter('test6.avi', fourcc, 100, (1280, 720))

    cap = cv2.VideoCapture(0) #Input live stream from camera
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    while cap.isOpened():
        start = time.time()
        ret,frame = cap.read()
        frame_number += 1
        #if not ret:
            #print("CAM NOT OPEND")
            #break

        frame, a, b, names = detect(frame, face_detector, face_encoder, encoding_dict)

        frame = procces(frame,a,b,names)

        # # # If input video
        #print("Writing frame {} / {}".format(frame_number, length))
        #output_movie.write(frame)
        end = time.time()
        print(end-start)
        cv2.imshow('camera', frame)
        #
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    cap.release()
    cv2.destroyAllWindows()


