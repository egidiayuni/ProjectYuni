from PyQt5 import QtCore

class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.dataset = None #type: pd.DataFrame()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return self.dataset.shape[0]

    def columnCount(self, parent=QtCore.QModelIndex()):
        return self.dataset.shape[1]

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.dataset.columns[section]
            else:
                return section
            
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return str(self.dataset.iloc[index.row()][index.column()])

class UjiTunggalListModel(QtCore.QAbstractListModel):
    def __init__(self, labels: list, parent=None):
        super().__init__(parent=parent)
        self.labels = labels
        for i in range(len(labels)):
            label = ''
            if labels[i] == 2:
                label = "Positive"
            elif labels[i] == 1:
                label = "Neutral"
            elif labels[i] == 0:
                label = 'Negative'
            
            labels[i] = label
                

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.labels)

    def data(self, QModelIndex, role=QtCore.Qt.DisplayRole):
        row = QModelIndex.row()
        if role == QtCore.Qt.DisplayRole:
            return 'model {}: {}'.format(row+1, self.labels[row])
            return 'model ' + (row+1) + ': ' + self.labels[row]

    


