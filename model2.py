import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QFileDialog, QPushButton, QVBoxLayout, QHBoxLayout, \
    QRadioButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir
import model



class AIApp(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(800, 600)

        #assuming python works like java, other methods should be able to access and modify this
        self.imageUploaded = False

        # Main container
        mainLayout = QVBoxLayout()

        # Radio buttons for model types
        layout = QVBoxLayout()
        radiobutton = QRadioButton('Flowers')
        radiobutton.type = 'Flowers'
        radiobutton.setChecked(True)
        radiobutton.toggled.connect(self.change_model)
        layout.addWidget(radiobutton)

        radiobutton = QRadioButton('Landscapes')
        radiobutton.type = 'Landscapes'
        radiobutton.toggled.connect(self.change_model)
        layout.addWidget(radiobutton)

        mainLayout.addLayout(layout)

        # set up and connect button 1
        self.button1 = QPushButton('Upload Image')
        self.button1.clicked.connect(self.get_image_file)

        # set up and connect button 2
        self.button2 = QPushButton('Check This Image')
        self.button2.clicked.connect(self.check_image)

        # labels to store image preview and instructions respectively
        self.imageLabel = QLabel()
        self.textLabel = QLabel()

        # add all widgets to the layout
        layout = QHBoxLayout()
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)

        mainLayout.addLayout(layout)
        mainLayout.addWidget(self.textLabel)
        mainLayout.addWidget(self.imageLabel)

        # the final layout should be the two buttons side by side at the top, then the text box below that, and the image preview at the bottom
        self.setLayout(mainLayout)

        # initializes text of the text box
        self.textLabel.setText('upload an image to check what kind of landscape it is')

        # set up for prediction
        self.im_path = None
        self.m = model.flower_model()

    def change_model(self):
        """Switches between the flower and landscape classifier

        """
        radioButton = self.sender()
        if radioButton.isChecked():
            if radioButton.type == 'Flowers':
                self.m = model.flower_model()
            else:
                self.m = model.landscape_model()

    def get_image_file(self):
        """Opens the file explorer and allows the user to select any of the given image files.

        """
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', './',
                                                   "Image files (*.jpg *.jpeg *.gif *.png)")
        self.imageLabel.setPixmap(QPixmap(file_name))
        self.im_path = file_name
        self.textLabel.setText('check image to see what kind of landscape it is')
        self.imageUploaded = True

    def check_image(self):
        """Sends the selected image to the image classifier and outputs the result

        """
        if (self.imageUploaded):
            s = self.m.predict(self.im_path)
            self.textLabel.setText(s)
        else:
            self.textLabel.setText('you have to upload an image before you check it')


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = AIApp()
    window.show()

    sys.exit(app.exec_())