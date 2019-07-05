# -*- coding: utf-8 -*-
# Icon made by Freepik from www.flaticon.com

import sys
from PyQt5 import QtCore
from dialog import *

if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUI()

        self.file_path = None

    def setupUI(self):
        self.resize(440, 240)

        icon = QIcon()
        icon.addPixmap(QPixmap("../../Desktop/CSS497/Program2/analysis.ico"), QIcon.Normal, QIcon.Off)
        self.setWindowIcon(icon)

        self.horizontalLayoutWidget = QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QRect(20, 40, 400, 80))
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(self.horizontalLayoutWidget)
        font = QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.horizontalLayout.addWidget(self.label)

        self.lineEdit = QLineEdit(self.horizontalLayoutWidget)
        font = QFont()
        font.setPointSize(18)
        self.lineEdit.setFont(font)
        self.horizontalLayout.addWidget(self.lineEdit)

        self.toolButton = QToolButton(self.horizontalLayoutWidget)
        font = QFont()
        font.setPointSize(18)
        self.toolButton.setFont(font)
        self.toolButton.clicked.connect(self.toolButton_clicked)
        self.horizontalLayout.addWidget(self.toolButton)

        self.horizontalLayoutWidget_2 = QWidget(self)
        self.horizontalLayoutWidget_2.setGeometry(QRect(150, 110, 270, 80))
        self.horizontalLayout_2 = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)

        self.exitPushButton = QPushButton(self.horizontalLayoutWidget_2)
        font = QFont()
        font.setPointSize(14)
        self.exitPushButton.setFont(font)
        self.horizontalLayout_2.addWidget(self.exitPushButton)
        self.exitPushButton.clicked.connect(self.close)

        self.okPushButton = QPushButton(self.horizontalLayoutWidget_2)
        font = QFont()
        font.setPointSize(14)
        self.okPushButton.setFont(font)
        self.horizontalLayout_2.addWidget(self.okPushButton)
        self.okPushButton.clicked.connect(self.okButton_clicked)

        self.retranslateUi(self)
        QMetaObject.connectSlotsByName(self)

    def toolButton_clicked(self):
        fname = QFileDialog.getOpenFileName(self)
        self.lineEdit.setText(fname[0])

    @QtCore.pyqtSlot()
    def okButton_clicked(self):
        self.file_path = self.lineEdit.text()
        dig = Dialog(self.file_path)
        dig.exec_()

    def retranslateUi(self, Dialog):
        _translate = QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "KIM LAB"))
        self.label.setText(_translate("Dialog", "File:"))
        self.toolButton.setText(_translate("Dialog", "..."))
        self.exitPushButton.setText(_translate("Dialog", "Exit"))
        self.okPushButton.setText(_translate("Dialog", "OK"))


# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook

def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # app.exec_()
    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")