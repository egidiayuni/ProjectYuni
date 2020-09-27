from PyQt5 import QtCore, QtWidgets, QtGui
from mainwindowui import Ui_MainWindow

from models import TableModel
from nb import runNB
import pandas as pd
from dialogs import UjiDataTunggalDialog

from sklearn.model_selection import train_test_split
import re

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, flags=QtCore.Qt.WindowFlags()):
        super().__init__(parent, flags)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pilihButton.clicked.connect(self.onPilihButton)
        self.ui.okButton.clicked.connect(self.onOkButton)
        self.ui.ujiDataTunggalButton.clicked.connect(self.onUjiDataTunggal)
        self.ui.ujiDataTunggalButton.setEnabled(False)

        self.ui.cleanButton.clicked.connect(self.onCleanButton)

        self.data_list=[]
        self.dataset = None #type: pd.DataFrame

        #
        pixmap = QtGui.QPixmap(R'usd.png')
        self.ui.label_3.setPixmap(pixmap.scaled(self.ui.label_3.width(), self.ui.label_3.height()))

        
    def onPilihButton(self):
        file_name, ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Open', QtCore.QDir.homePath(), 'CSV(*.csv)')

        if file_name == '':
            return

        self.ui.lineEdit.setText(file_name)
        self.dataset = pd.read_csv(file_name)
        
        tableModel = TableModel(self)
        tableModel.dataset = self.dataset
        self.ui.tableView.setModel(tableModel)

        # self.X, self.target, self.vectorizer = runPreprocessing(dataset)

    def onOkButton(self):
        n_folds = int(self.ui.kfoldLineEdit.text())
        akurasi, self.data_list = runNB(self.dataset, n_folds=2)
        self.ui.akuasiLineEdit.setText(str(akurasi))
        self.ui.ujiDataTunggalButton.setEnabled(True)

    def onUjiDataTunggal(self):
        dialog = UjiDataTunggalDialog(self)
        dialog.data_list = self.data_list
        dialog.exec()

    def onCleanButton(self):
        row_count = self.dataset.shape[0]

        for i in range(row_count):
            cleaned_text = self.clean_tweet(self.dataset.at[i, 'text'])
            self.dataset.at[i, 'text'] = cleaned_text
        # endfor

        model = self.ui.tableView.model() #type: TableModel
        model.dataChanged.emit(model.index(0,1), model.index(row_count,1))


    def clean_tweet(self, tweet):
        cleanedtweet = ' '.join(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+'')", " ", tweet).split())
        return cleanedtweet

    
        