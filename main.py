import sys
from PyQt5.QtWidgets import QApplication, QWidget
from QtMainWindow import QtMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets

#设定哪个UI为主窗体 
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
   
    mainwindows=QtMainWindow()
    mainwindows.show()
    sys.exit(app.exec_())

