# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1035, 547)
        self.titleLabel = QtWidgets.QLabel(MainWindow)
        self.titleLabel.setGeometry(QtCore.QRect(380, 10, 251, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.titleLabel.setFont(font)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.titleLabel.setObjectName("titleLabel")
        self.label_3 = QtWidgets.QLabel(MainWindow)
        self.label_3.setGeometry(QtCore.QRect(280, 10, 61, 61))
        self.label_3.setObjectName("label_3")
        self.widget = QtWidgets.QWidget(MainWindow)
        self.widget.setGeometry(QtCore.QRect(180, 90, 671, 411))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.pilihButton = QtWidgets.QPushButton(self.widget)
        self.pilihButton.setObjectName("pilihButton")
        self.horizontalLayout.addWidget(self.pilihButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tableView = QtWidgets.QTableView(self.widget)
        self.tableView.setObjectName("tableView")
        self.horizontalLayout_2.addWidget(self.tableView)
        self.cleanButton = QtWidgets.QPushButton(self.widget)
        self.cleanButton.setObjectName("cleanButton")
        self.horizontalLayout_2.addWidget(self.cleanButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.kfoldLineEdit = QtWidgets.QLineEdit(self.widget)
        self.kfoldLineEdit.setObjectName("kfoldLineEdit")
        self.horizontalLayout_3.addWidget(self.kfoldLineEdit)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.okButton = QtWidgets.QPushButton(self.widget)
        self.okButton.setObjectName("okButton")
        self.gridLayout.addWidget(self.okButton, 0, 1, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.akuasiLineEdit = QtWidgets.QLineEdit(self.widget)
        self.akuasiLineEdit.setObjectName("akuasiLineEdit")
        self.horizontalLayout_4.addWidget(self.akuasiLineEdit)
        self.gridLayout.addLayout(self.horizontalLayout_4, 1, 0, 1, 1)
        self.ujiDataTunggalButton = QtWidgets.QPushButton(self.widget)
        self.ujiDataTunggalButton.setObjectName("ujiDataTunggalButton")
        self.gridLayout.addWidget(self.ujiDataTunggalButton, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Form"))
        self.titleLabel.setText(_translate("MainWindow", "Klasifikasi Sentimen\n"
" Tweet terhadap Covid-19"))
        self.label_3.setText(_translate("MainWindow", "Logo"))
        self.pilihButton.setText(_translate("MainWindow", "Pilih"))
        self.cleanButton.setText(_translate("MainWindow", "Clean"))
        self.label.setText(_translate("MainWindow", "K-Fold"))
        self.okButton.setText(_translate("MainWindow", "OK"))
        self.label_2.setText(_translate("MainWindow", "Akurasi"))
        self.ujiDataTunggalButton.setText(_translate("MainWindow", "Uji Data Tunggal"))
