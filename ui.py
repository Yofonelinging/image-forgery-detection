import sys
import os

# Prevent TensorFlow/Qt DLL conflict crash on Windows (exit code -1073741819 / 0xC0000005)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU, avoids GPU DLL conflicts
os.environ["OMP_NUM_THREADS"] = "1"

# CRITICAL: Import TensorFlow/Keras BEFORE PyQt5 on Windows to avoid DLL conflict crash
from ela import convert_to_ela_image
from prediction import predict_result

# Now it's safe to import PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QLabel, QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt



class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("gui.ui", self)
        self.Browse.clicked.connect(self.open_image)
        self.Test.clicked.connect(self.result)
        self.Quit.clicked.connect(self.close_main_window)

    def open_image(self):
        # display original image
        self.fname = QFileDialog.getOpenFileName(
            self, "Open file", "C:'", ("*.png, *.xmp *.jpg")
        )
        self.filename.setText(self.fname[0])
        pixmap = QPixmap(self.fname[0])
        self.ORIGINAL_IMAGE.setPixmap(pixmap)
        self.ORIGINAL_IMAGE.setPixmap(
            pixmap.scaled(self.ORIGINAL_IMAGE.size(), Qt.IgnoreAspectRatio)
        )
        self.ORIGINAL_IMAGE.show()

        # display ela image
        convert_to_ela_image(self.fname[0], 90)
        pixmap1 = QPixmap("ela_image.png")
        self.ELA_IMAGE.setPixmap(pixmap1)
        self.ELA_IMAGE.setPixmap(
            pixmap1.scaled(self.ELA_IMAGE.size(), Qt.IgnoreAspectRatio)
        )
        self.ELA_IMAGE.show()

    def result(self):
        (prediction, confidence) = predict_result((self.fname))
        self.Result.setText(f"Prediction: {prediction}\nConfidence: {confidence} %")

    def close_main_window(self):
        # quit window
        reply = QMessageBox.question(
            self,
            "Quit",
            "Are you sure you want to quit?",
            QMessageBox.Cancel | QMessageBox.Close,
        )
        if reply == QMessageBox.Close:
            sys.exit()


def main():
    app = QApplication(sys.argv)
    Result = MainWindow()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(Result)
    widget.setFixedWidth(620)
    widget.setFixedHeight(560)
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
