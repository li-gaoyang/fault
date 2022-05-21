# -*- coding: utf-8 -*-
from re import L
from typing_extensions import Self
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog
from matplotlib.pyplot import close
from Ui_QtMainWindow import Ui_MainWindow
from PyQt5.QtCore import QFileInfo
from PyQt5.QtWidgets import QFileDialog
import os
from ftplib import FTP
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import uuid
import sys
import time
import win32api
import threading
from multiprocessing import Process
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import fault
import readsgyvtk
import cv2 as cv 

from skimage.morphology import skeletonize,thin,medial_axis

class QtMainWindow(QMainWindow):
    """主窗体窗体"""

    def __init__(self, parent=None):
        super(QtMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        
        self.ui.setupUi(self)
        self.initUI()
    

    def initUI(self):
        self.setWindowTitle('PyQt5 fault detection')
        
      

        self.ui.btButton.clicked.connect(self.btclick)
        self.ui.lineEdit_path.setText("demo01.sgy")
        self.ui.lineEdit_ker1.setText("25")
        self.ui.lineEdit_ker2.setText("5")
        self.ui.lineEdit_minconn.setText("15")
        self.ui.lineEdit_minrem.setText("40")

        self.show()

  
    def btclick(self):
        filepath=str(self.ui.lineEdit_path.text())
        #读取地震数据
        #sesmicdt=readsgyvtk.load_data_from_sgy("demo01.sgy")
        sesmicdt=readsgyvtk.load_data_from_sgy(filepath)
        # sesmicdt=cv.cvtColor(sesmicdt,cv.COLOR_BGR2GRAY)

        #计算方向场
        ker1=int(self.ui.lineEdit_ker1.text())
        ker2=int(self.ui.lineEdit_ker2.text())
        matCoh2=fault.Cal_Direction_Field(sesmicdt,ker1,ker2)
      
        
        print("方向场卷积核:"+str(ker1)+"X"+str(ker2))
       
        #二值化
      
        thresh1,thresholdval=fault.binfun(matCoh2)
       
        print("二值化阈值:"+str(thresholdval))
        
        #闭操作
        closeding=fault.closed(thresh1,ker1/10,1)
        
        print("闭操作:"+str(ker1)+"X"+str(ker2))
      
        #zhang-suen
        # cv.imwrite("Direction_Field_binary_closed2.jpg",closeding)
        # closeding=cv.imread("Direction_Field_binary_closed2.jpg")
        # mid_points=zhang_suen.Zhang_Suen_thining(closeding)
        # mid_points2=closeding.copy()
        # w,h=mid_points.shape
        # for i in range(w):
        #     for j in range(h):
        #         if mid_points[i][j]!=255:
        #             mid_points2[i][j]=0
        # cv.imwrite("zhang_suen.jpg",mid_points2)
        # return

        #lee
        # w,h=closeding.shape
        # for i in range(w):
        #     for j in range(h):
        #         if closeding[i][j]!=0:
        #             closeding[i][j]=1
        # mid_points=skeletonize(closeding)
        # mid_points2=closeding.copy()
        # for i in range(w):
        #     for j in range(h):
        #         if mid_points[i][j]==False:
        #             mid_points2[i][j]=0
        #         else:
        #             mid_points2[i][j]=255
        
        # cv.imwrite("lee.jpg",mid_points2)
        #return

        #中轴骨架化
        # mid_points=medial_axis(closeding)
        # w,h=closeding.shape
        # mid_points2=closeding.copy()
        # for i in range(w):
        #     for j in range(h):
        #         if mid_points[i][j]==False:
        #             mid_points2[i][j]=0
        #         else:
        #             mid_points2[i][j]=255
        
        # cv.imwrite("medial_axis.jpg",mid_points2)
        # return

        #提取中心点
        mid_points=fault.extmidpoints(closeding)
        
        print("提取中心点完成！")
      
        

        #连接相邻的线段
        connval=int(self.ui.lineEdit_minconn.text())
        mid_points_con=fault.connectlines(mid_points,connval)
        
         #剔除线段长度小于40的线段
        minremval=int(self.ui.lineEdit_minrem.text())
        result_img=fault.removeshortlines(mid_points_con,minremval)

        print("最小连接相邻点间距:"+str(connval))
         
      
        
        print("剔除最小线段:"+str(minremval))

        fault.getresultimg(result_img)
        print("断层识别完成！")


        return