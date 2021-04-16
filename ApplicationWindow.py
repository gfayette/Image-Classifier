import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QFileDialog, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir
#TODO import project script here


class AIApp(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(800, 600)

        #set up and connect button 1
        self.button1 = QPushButton('Upload Image')
        self.button1.clicked.connect(self.get_image_file)

        #set up and connect button 2
        self.button2 = QPushButton('Check This Image')
        self.button2.clicked.connect(self.checkImage)



        #labels to store image preview and instructions respectively
        self.imageLabel = QLabel()
        self.textLabel = QLabel()

        #add all widgets to the layout
        layout = QHBoxLayout()
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)

        labelLayout = QVBoxLayout()
        labelLayout.addLayout(layout)
        labelLayout.addWidget(self.textLabel)
        labelLayout.addWidget(self.imageLabel)

        #the final layout should be the two buttons side by side at the top, then the text box below that, and the image preview at the bottom
        self.setLayout(labelLayout)

        #initializes text of the text box
        self.textLabel.setText('upload an image to check what kind of landscape it is')

    def get_image_file(self):
        #the './' parameter might be a source of an error. That parameter is the directory that opens in the fileDialog, and I want it to remain default
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', './', "Image files (*.jpg *.jpeg *.gif *.png)")
        self.imageLabel.setPixmap(QPixmap(file_name))
        self.textLabel.setText('check image to see what kind of landscape it is')

    def checkImage(self):
        #TODO call image classifier from this function. It would be pretty easy if you just need the filename for the script to access the image, but it shouldn't be too hard.
        self.textLabel.setText('this text is supposed to be the output from the image classifier')
        



if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = AIApp()
    window.show()

    sys.exit(app.exec_())

