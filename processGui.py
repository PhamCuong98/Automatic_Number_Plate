from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import (QDialog ,QApplication,QWidget, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout)
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
from process_AI import process_img
from process_MySQL import mySQL
from datetime import datetime
class processCamera(QWidget):
    def __init__(self, npImage):
        super().__init__()
        self.np_Image= npImage
        self.show()

    def getCamera(self, np_Image):
        mydialog= QHBoxLayout(self)
        qimage = QImage(np_Image, np_Image.shape[1], np_Image.shape[0], QImage.Format_RGB888)                                                                                                                                                 
                     
        image = QPixmap(qimage)
        label = QLabel(self)
        label.setPixmap(image)
        label.resize(600,300)
        mydialog.addWidget(label)
        print(np_Image)
        
class processImage(QWidget):
    def __init__(self, path_img):
        super().__init__()
        self.path_img= path_img
        self.show()

    def getImage(self, path_img):
        self.infor= []
        mypic= QHBoxLayout(self)
        pic = QPixmap(path_img)
        label= QLabel(self)
        label.setPixmap(pic)
        label.resize(300,200)
        mypic.addWidget(label)

        self.process= process_img(path_img)
        license, result_arr = self.process.process_AI(path_img)
        print(license)

        qimage = QImage(result_arr, result_arr.shape[1], result_arr.shape[0], QImage.Format_RGB888)                                                                                                                                                 
                     
        yolo = QPixmap(qimage)
        label_yolo = QLabel(self)
        label_yolo.setPixmap(yolo)
        label_yolo.resize(300,200)
        mypic.addWidget(label_yolo)
        
        time, day= self.getTime()
        print(time)
        print(day)
        self.infor.append(time)
        self.infor.append(day)
        self.infor.append(license)
        sql= mySQL(self.infor)
        sql.public()
    def getTime(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_day = now.strftime("%d/%m/%Y")
        return current_time, current_day
        
if __name__ == '__main__':
    main()