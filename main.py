from mainwindow import MainWindow
from PyQt5 import QtWidgets

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())