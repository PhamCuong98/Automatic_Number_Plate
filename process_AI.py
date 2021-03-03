import os
import colorsys
import sys
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.compat.v1.keras.models import load_model

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)

print(tf.__version__)
model = load_model(r"Model/my_model.h5")
#model.summary()

class Yolo4(object):
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.sess = K.get_session()

        # Load model, or construct model and load weights.
        self.yolo4_model = yolo4_body(Input(shape=(608, 608, 3)), num_anchors//3, num_classes)
        self.yolo4_model.load_weights(model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        if self.gpu_num>=2:
            self.yolo4_model = multi_gpu_model(self.yolo4_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo4_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score)

    def __init__(self, score, iou, anchors_path, classes_path, model_path, gpu_num=1):
        self.score = score
        self.iou = iou
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.load_yolo()

    def close_session(self):
        self.sess.close()

    def detect_image(self, image, model_image_size=(608, 608)):
        start = timer()

        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image, top, left, bottom, right

class process_img(object):
    def __init__(self, image):
        self.image= image

    def detect_box(self, input_size=(608, 608)):
        #image_arr= np.array(self.image)
        result, top, left, bottom, right = yolo4_model.detect_image(self.image, model_image_size=model_image_size)
        return result, top, left, bottom, right

    def dectect_character(self):
        self.image_arr= np.array(self.image)
        result, top, left, bottom, right= self.detect_box()
        
        pts1= np.float32([[left, top], [right,top], [left, bottom], [right, bottom]])
        pts2= np.float32([[0, 0], [600,0], [0, 300], [600, 300]])
        img_bird= cv2.getPerspectiveTransform(pts1, pts2)
        result_bird= cv2.warpPerspective(self.image_arr, img_bird, (600, 300))
        image_split= cv2.cvtColor(result_bird, cv2.COLOR_BGR2GRAY)
        ret, thresh2= cv2.threshold(image_split, 125, 255, cv2.THRESH_BINARY)
        cv2.floodFill(thresh2, None, (0,0), 255)
        thresh_blur = cv2.medianBlur(thresh2, 5)
        thresh2_2 = cv2.bitwise_not(thresh_blur)
        contours, hierarchy = cv2.findContours(thresh2_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        countContours = 0
        box=[]
        for contour in contours:
            x, y, w, h = contourRect = cv2.boundingRect(contour)
            ratio= h/w
            if 1.8<=ratio<=5.5:
                if 2< thresh2_2.shape[0]/h< 3.5:
                    print("Trong so")
                    print(x, y, w, h, thresh2_2.shape[0])
                    countContours += 1
                    #cv2.rectangle(result_bird, (x, y), (x + w, y + h), (0, 255, 0))
                    box_img= thresh2_2[y:y+h,x:x+w]
                    box.append(box_img)
        print("So contour tim dc: ", countContours)
        return result, box, countContours, thresh2_2

    def process_AI(self, image):
        img_detect= process_img(image)
        result, boxes, countContours, result_bird= img_detect.dectect_character()
        
        result_arr= np.array(result_bird)
        num=[]
        licenses=[]
        label_data= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C','D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
        #xu ly tung box chu so 
        for i in range(len(boxes)):
            print(boxes[i].shape)
            box_img= cv2.resize(boxes[i], (28,28))

            box_img_3=np.stack((box_img,)*3, -1)
            test= box_img_3.reshape(1,28,28,3)
            predict= model.predict(test)
            value= np.argmax(predict)
            if value <31:
                num.append(label_data[value])
            licenses = " ".join(num)
        return licenses, result_arr

# Duong dan den file h5
model_path = 'Model/yolo4_weight.h5'
    # File anchors cua YOLO
anchors_path = 'Model/yolo4_anchors.txt'
    # File danh sach cac class
classes_path = 'Model/yolo.names'

score = 0.5
iou = 0.5
model_image_size = (608, 608)

yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path) #Tao doi truong Yolo

if __name__== '__main__':
    main()

