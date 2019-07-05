from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from program import *


if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


class Dialog(QDialog):
    def __init__(self, file_path, parent=None):
        super(Dialog, self).__init__(parent)
        self.setupUI()

        self.file_path = file_path
        self.session_type = None

    def setupUI(self):
        self.resize(480, 340)

        icon = QIcon()
        icon.addPixmap(QPixmap("../../Desktop/CSS497/Program2/analysis.ico"), QIcon.Normal, QIcon.Off)
        self.setWindowIcon(icon)

        self.layoutWidget = QWidget(self)
        self.layoutWidget.setGeometry(QRect(60, 210, 380, 26))
        self.horizontalLayout_3 = QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)

        self.troughLowerDoubleSpinBox = QDoubleSpinBox(self.layoutWidget)
        font = QFont()
        font.setPointSize(18)
        self.troughLowerDoubleSpinBox.setFont(font)
        self.troughLowerDoubleSpinBox.setRange(-100, 100)
        self.horizontalLayout_3.addWidget(self.troughLowerDoubleSpinBox)

        self.troughPositionLabel = QLabel(self.layoutWidget)
        font = QFont()
        font.setPointSize(18)
        self.troughPositionLabel.setFont(font)
        self.horizontalLayout_3.addWidget(self.troughPositionLabel)

        self.troughUpperDoubleSpinBox = QDoubleSpinBox(self.layoutWidget)
        font = QFont()
        font.setPointSize(18)
        self.troughUpperDoubleSpinBox.setFont(font)
        self.troughUpperDoubleSpinBox.setRange(-100, 100)
        self.horizontalLayout_3.addWidget(self.troughUpperDoubleSpinBox)

        self.widget = QWidget(self)
        self.widget.setGeometry(QRect(170, 270, 280, 38))
        self.horizontalLayout_4 = QHBoxLayout(self.widget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)

        self.exitPushButton = QPushButton(self.widget)
        font = QFont()
        font.setPointSize(14)
        self.exitPushButton.setFont(font)
        self.horizontalLayout_4.addWidget(self.exitPushButton)
        self.exitPushButton.clicked.connect(self.close)

        self.okPushButton = QPushButton(self.widget)
        font = QFont()
        font.setPointSize(14)
        self.okPushButton.setFont(font)
        self.horizontalLayout_4.addWidget(self.okPushButton)
        self.okPushButton.clicked.connect(self.ok_clicked)

        self.widget1 = QWidget(self)
        self.widget1.setGeometry(QRect(60, 130, 380, 26))
        self.horizontalLayout_2 = QHBoxLayout(self.widget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)

        self.peakLowerDoubleSpinBox = QDoubleSpinBox(self.widget1)
        font = QFont()
        font.setPointSize(18)
        self.peakLowerDoubleSpinBox.setFont(font)
        self.peakLowerDoubleSpinBox.setRange(-100, 100)
        self.horizontalLayout_2.addWidget(self.peakLowerDoubleSpinBox)

        self.peakPositionLabel = QLabel(self.widget1)
        font = QFont()
        font.setPointSize(18)
        self.peakPositionLabel.setFont(font)
        self.horizontalLayout_2.addWidget(self.peakPositionLabel)

        self.peakUpperDoubleSpinBox = QDoubleSpinBox(self.widget1)
        font = QFont()
        font.setPointSize(18)
        self.peakUpperDoubleSpinBox.setFont(font)
        self.peakUpperDoubleSpinBox.setRange(-100, 100)
        self.horizontalLayout_2.addWidget(self.peakUpperDoubleSpinBox)

        self.widget2 = QWidget(self)
        self.widget2.setGeometry(QRect(40, 30, 210, 30))
        self.horizontalLayout = QHBoxLayout(self.widget2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(self.widget2)
        font = QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.horizontalLayout.addWidget(self.label)

        self.comboBox = QComboBox(self.widget2)
        font = QFont()
        font.setPointSize(18)
        self.comboBox.setFont(font)
        self.comboBox.addItem('')
        self.comboBox.addItem('Session 1')
        self.comboBox.addItem('Session 2')
        self.comboBox.addItem('Session 3')
        self.comboBox.addItem('Session 4')
        self.comboBox.addItem('Session 5')
        # self.comboBox.activated[str].connect(self.onActivated)
        self.horizontalLayout.addWidget(self.comboBox)

        self.widget3 = QWidget(self)
        self.widget3.setGeometry(QRect(40, 90, 250, 30))
        self.horizontalLayout_5 = QHBoxLayout(self.widget3)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)

        self.peakCheckBox = QCheckBox(self.widget3)
        font = QFont()
        font.setPointSize(18)
        self.peakCheckBox.setFont(font)
        self.horizontalLayout_5.addWidget(self.peakCheckBox)

        self.peakDoubleSpinBox = QDoubleSpinBox(self.widget3)
        font = QFont()
        font.setPointSize(18)
        self.peakDoubleSpinBox.setFont(font)
        self.peakDoubleSpinBox.setRange(-100, 100)
        self.horizontalLayout_5.addWidget(self.peakDoubleSpinBox)

        self.widget4 = QWidget(self)
        self.widget4.setGeometry(QRect(40, 170, 260, 30))
        self.horizontalLayout_6 = QHBoxLayout(self.widget4)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)

        self.troughCheckBox = QCheckBox(self.widget4)
        font = QFont()
        font.setPointSize(18)
        self.troughCheckBox.setFont(font)
        self.horizontalLayout_6.addWidget(self.troughCheckBox)

        self.troughDoubleSpinBox = QDoubleSpinBox(self.widget4)
        font = QFont()
        font.setPointSize(18)
        self.troughDoubleSpinBox.setFont(font)
        self.troughDoubleSpinBox.setRange(-100, 100)
        self.horizontalLayout_6.addWidget(self.troughDoubleSpinBox)

        self.retranslateUi(self)
        QMetaObject.connectSlotsByName(self)
        self.setTabOrder(self.comboBox, self.peakCheckBox)
        self.setTabOrder(self.peakCheckBox, self.peakDoubleSpinBox)
        self.setTabOrder(self.peakDoubleSpinBox, self.peakLowerDoubleSpinBox)
        self.setTabOrder(self.peakLowerDoubleSpinBox, self.peakUpperDoubleSpinBox)
        self.setTabOrder(self.peakUpperDoubleSpinBox, self.troughCheckBox)
        self.setTabOrder(self.troughCheckBox, self.troughDoubleSpinBox)
        self.setTabOrder(self.troughDoubleSpinBox, self.troughLowerDoubleSpinBox)
        self.setTabOrder(self.troughLowerDoubleSpinBox, self.troughUpperDoubleSpinBox)
        self.setTabOrder(self.troughUpperDoubleSpinBox, self.exitPushButton)
        self.setTabOrder(self.exitPushButton, self.okPushButton)

    def ok_clicked(self):
        df = Program.read_file(self.file_path)
        session_type = str(self.comboBox.currentText())

        if self.peakCheckBox.isChecked():
            new_df = Program.select_session(df, session_type, self.peakDoubleSpinBox.value(),
                                            self.peakLowerDoubleSpinBox.value(), self.peakUpperDoubleSpinBox.value(),
                                            None, None, None)
        if self.troughCheckBox.isChecked():
            new_df = Program.select_session(df, session_type, None, None, None, self.troughDoubleSpinBox.value(),
                                      self.troughLowerDoubleSpinBox.value(), self.troughUpperDoubleSpinBox.value())
        new_df.to_excel('Data.xlsx', sheet_name='All')

        stand_df = {}
        for i in range(new_df.index.get_level_values(0).nunique()):
            if i % 2 == 0:
                stand_df[i] = Program.standardization(new_df.loc[i], new_df.loc[i + 1, 3], new_df.loc[i + 1, 4])
        stand_df = pd.concat({k: pd.DataFrame(v) for k, v in stand_df.items()}, axis=0)
        stand_df.to_excel('Standard Data.xlsx', sheet_name='All')

        mean_sem = stand_df.mean(axis=1), stand_df.sem(axis=1)
        temp_df = pd.DataFrame(list(mean_sem)).T

        Program.graph(temp_df)

        self.close()

    def retranslateUi(self, Dialog):
        _translate = QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Condition"))
        self.troughPositionLabel.setText(_translate("Dialog", "< Trough Position <"))
        self.exitPushButton.setText(_translate("Dialog", "Exit"))
        self.okPushButton.setText(_translate("Dialog", "OK"))
        self.peakPositionLabel.setText(_translate("Dialog", "< Peak Position <"))
        self.label.setText(_translate("Dialog", "Session:"))
        self.peakCheckBox.setText(_translate("Dialog", "Peak Z-score >"))
        self.troughCheckBox.setText(_translate("Dialog", "Trough Z-score <"))