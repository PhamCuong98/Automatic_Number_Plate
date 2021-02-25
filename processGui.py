from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import (QDialog ,QMessageBox, QApplication,QWidget, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton)
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
        self.initUI()
    
    def initUI(self):
        self.show()

    def getImage(self, path_img):
        self.setGeometry( 300, 300, 350, 300 )
        self.setWindowTitle( 'Review' )
        self.show()
        process= process_img(path_img)
        licenses, result_arr = process.process_AI(path_img)
        print(licenses)

        self.horizontalLayout = QHBoxLayout()
        pic = QPixmap(path_img)
        label= QLabel()
        label.setPixmap(pic)
        label.resize(300,200)
        self.horizontalLayout.addWidget(label)
        qimage = QImage(result_arr, result_arr.shape[1], result_arr.shape[0], QImage.Format_RGB888)                                                                                                                                                                  
        yolo = QPixmap(qimage)
        label_yolo = QLabel()
        label_yolo.setPixmap(yolo)
        label_yolo.resize(300,200)
        self.horizontalLayout.addWidget(label_yolo)

        self.horizontalLayout2 = QHBoxLayout()
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(lambda:self.exit())
        send = QPushButton('Send')
        send.clicked.connect(lambda:self.sendMySQL(licenses))
        self.horizontalLayout2.addWidget(cancel)
        self.horizontalLayout2.addWidget(send)

        self.verticalLayout = QVBoxLayout()
        text1= QLabel("Ket qua")
        text1.move(200,200)
        text2= QLabel(licenses)
        text2.move(250,200)
        self.verticalLayout.addWidget(text1)
        self.verticalLayout.addWidget(text2)

        self.verticalEnd = QVBoxLayout(self)
        self.verticalEnd.addLayout(self.horizontalLayout)
        self.verticalEnd.addLayout(self.horizontalLayout2)
        self.verticalEnd.addLayout(self.verticalLayout)
        self.setLayout(self.verticalEnd)

    def exit(self):
        print("PhamCuong_1st")
        reply = QMessageBox.question(
            self, "Message",
            "Are you sure you want to quit? Any unsaved work will be lost.",
            QMessageBox.Close | QMessageBox.Cancel)

        if reply == QMessageBox.Close:
            self.close()
        else:
            pass

    def sendMySQL(self, licenses):
        infor= []
        time, day= self.getTime()
        print(time)
        print(day)
        infor.append(time)
        infor.append(day)
        infor.append(licenses)
        sql= mySQL(infor)
        sql.public()
        self.exit()
        
    def getTime(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_day = now.strftime("%d/%m/%Y")
        return current_time, current_day


if __name__ == '__main__':
    main()