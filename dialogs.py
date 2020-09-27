from PyQt5 import QtCore, QtWidgets, QtGui
from ujidatatunggal_ui import Ui_UjiDataTunggalDialog
from models import UjiTunggalListModel

class UjiDataTunggalDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, flags=QtCore.Qt.WindowFlags()):
        super().__init__(parent=parent, flags=flags)

        self.data_list = []

        self.ui = Ui_UjiDataTunggalDialog()
        self.ui.setupUi(self)

        self.ui.okButton.clicked.connect(self.onOkButton)
        self.ui.clearButton.clicked.connect(self.onClearButton)


    def onOkButton(self):
        text = self.ui.textEdit.toPlainText()
        labels = []
        for i in self.data_list:
            vectorizer = i['vectorizer']
            test = vectorizer.transform([text])
            MNB = i['MNB']
            predict = MNB.predict(test)
            labels.append(predict[0])
            print(predict)
        
        model = UjiTunggalListModel(labels, self)
        self.ui.listView.setModel(model)

    def onClearButton(self):
        self.ui.textEdit.clear()

