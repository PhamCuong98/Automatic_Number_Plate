from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import (QHBoxLayout, QDialog ,QApplication,QPushButton,QWidget, QMainWindow, 
QMessageBox, QLineEdit,QTabWidget, QLabel, QAction, QVBoxLayout, QGroupBox, QFileDialog)

import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from processGui import processCamera, processImage
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self): 
        super().__init__()
        self._run_flag = True

    def run(self):

        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)

        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title= "Detect license plate"
        self.left= 10
        self.top= 10
        self.height= 768
        self.width= 1024
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        mainMenu = self.menuBar()
        exit_UI= mainMenu.addMenu('Exit')
        self.exit_Button(exit_UI)
        
        self.table_widget= MyTableWidge(self)
        self.setCentralWidget(self.table_widget)
        
        self.show()
    def exit_Button(self, UI):
        exit_Button= QAction('Exit', self)
        exit_Button.setShortcut('Ctrl+Q')
        exit_Button.setStatusTip('Exit.application')
        exit_Button.triggered.connect(self.close)
        UI.addAction(exit_Button)

class MyTableWidge(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.disply_width = 600
        self.display_height = 900

        self.layout= QVBoxLayout(self) #Layout chinh cua Widget

        self.show()

        #Tao tab me co 2 tab con
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(self.disply_width, self.display_height)

        # Them 2 tab con
        self.tabs.addTab(self.tab1, "Camera")
        self.tabs.addTab(self.tab2, "Image")

    #<-------------------------Setup cho tab 1------------------------------>
        self.tab1.layout = QVBoxLayout(self) # Chuan hoa layout tab1

        #Tao các extension cho tab 1
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.text_label = QLabel('Webcam')

        self.pushbutton1Tab1 = QPushButton("OpenCamera")
        self.pushbutton1Tab1.clicked.connect(self.openCamera)

        self.pushbutton2Tab1= QPushButton("Detect")
        self.pushbutton2Tab1.clicked.connect(self.choose_img)

        self.pushbutton3Tab1= QPushButton("CloseCamera")
        self.pushbutton3Tab1.clicked.connect(self.closeEvent)

        #Add các extension vào tab
        self.tab1.layout.addWidget(self.image_label)   
        self.tab1.layout.addWidget(self.text_label)
        self.tab1.layout.addWidget(self.pushbutton1Tab1)
        self.tab1.layout.addWidget(self.pushbutton2Tab1)
        self.tab1.layout.addWidget(self.pushbutton3Tab1)

        #Bước set layout cho tab 1
        self.tab1.setLayout(self.tab1.layout)
    #<-------------------------------End----------------------------------->

    #<--------------------------Setup cho tab 2---------------------------->
        self.tab2.layout= QVBoxLayout(self)
        #Tạo các extension cho tab 2
        self.pushbutton1Tab2= QPushButton("Them path Image")
        self.pushbutton1Tab2.clicked.connect(self.openImage)
        
        self.tab2.layout.addWidget(self.pushbutton1Tab2)

        self.tab2.setLayout(self.tab2.layout)
    #<-------------------------------End----------------------------------->
        # Add tabs to widget
        self.layout.addWidget(self.tabs) 
        self.setLayout(self.layout)

    #<----------------------------Func for tab1---------------------------->
    def closeEvent(self):
        print('PhamCuong1st')

    def choose_img(self):
        #self.click()
        self.thread.change_pixmap_signal.connect(self.get_img)
        self.thread.stop()
        
    def openCamera(self):
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def click(self):
        print("PhamCuong_1st")

    def get_img(self, cv_img):
        print(type(cv_img))
        self.process= processCamera(cv_img)
        self.process.getCamera(cv_img)
    #<----------------------------Func for tab 2--------------------------->
    def openImage(self):
        path_img= self.openFileNameDialog()
        print(path_img)
        self.process= processImage(path_img)
        self.process.getImage(path_img)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(type(fileName))
        return fileName
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p) 

def main():
    app= QApplication(sys.argv)
    ex= App()
    ex.show()
    sys.exit(app.exec_())

if __name__== '__main__':
    main()