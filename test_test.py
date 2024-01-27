import os
import smtplib
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import mail
import cv2
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import sys
from PyQt6.QtGui import QPixmap, QImage, QPainter, QIcon
from PIL import Image
import cv2
from merged import process_images
from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import BASNet
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

os.makedirs('original', exist_ok=True)
# os.makedirs('masked', exist_ok=True)
os.makedirs('latest_file', exist_ok=True)


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.initUI()
        self.bg_image_flag = 0
        self.frame_flag = 0
        self.bg_img_path = ""
        self.frame_path = ""
        self.img_counter = 1
        # self.background_img_idx = 0
        # self.frame_img_idx = 0
        # self.pixmap = QPixmap

    def initUI(self):
        self.setGeometry(100, 50, 791, 899)
        self.setWindowTitle("Virtual Photo Studio")
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 741, 251))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.groupBox.setTitle("Background Image")
        self.browseBgImageButton = QtWidgets.QPushButton(parent=self.groupBox)
        self.browseBgImageButton.setGeometry(QtCore.QRect(50, 110, 241, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.browseBgImageButton.setFont(font)
        self.browseBgImageButton.setObjectName("browseBgImageButton")
        self.browseBgImageButton.setText("Select Background Image")
        self.browseBgImageButton.clicked.connect(self.browse_background_image)
        self.bgImageGraphicsView = QtWidgets.QGraphicsView(
            parent=self.groupBox)
        self.bgImageGraphicsView.setGeometry(QtCore.QRect(420, 31, 291, 201))
        self.bgImageGraphicsView.setObjectName("bgImageGraphicsView")

        self.groupBox_2 = QtWidgets.QGroupBox(self)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 270, 741, 241))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_2.setTitle("Frame")
        self.browseFrameButton = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.browseFrameButton.setGeometry(QtCore.QRect(70, 100, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.browseFrameButton.setFont(font)
        self.browseFrameButton.setObjectName("browseFrameButton")
        self.browseFrameButton.setText("Select Frame")
        self.browseFrameButton.clicked.connect(self.browse_frame)
        self.frameGraphicsView = QtWidgets.QGraphicsView(
            parent=self.groupBox_2)
        self.frameGraphicsView.setGeometry(QtCore.QRect(420, 26, 291, 201))
        self.frameGraphicsView.setObjectName("frameGraphicsView")

        self.groupBox_3 = QtWidgets.QGroupBox(self)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 520, 741, 331))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_2.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.groupBox_3.setTitle("Capture Photo")
        self.captureImageButton = QtWidgets.QPushButton(parent=self.groupBox_3)
        self.captureImageButton.setGeometry(QtCore.QRect(75, 100, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.captureImageButton.setFont(font)
        self.captureImageButton.setObjectName("captureImageButton")
        self.captureImageButton.setText("Capture Photo")
        self.captureImageButton.setDisabled(True)
        self.captureImageButton.clicked.connect(self.capture_image)
        self.outputImageGraphicsView = QtWidgets.QGraphicsView(
            parent=self.groupBox_3)
        self.outputImageGraphicsView.setGeometry(
            QtCore.QRect(420, 20, 291, 201))
        self.outputImageGraphicsView.setObjectName("outputImageGraphicsView")
        self.nameTextBox = QtWidgets.QLineEdit(parent=self.groupBox_3)
        self.nameTextBox.setGeometry(QtCore.QRect(15, 280, 211, 20))
        self.nameTextBox.setObjectName("nameTextBox")
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_3.setGeometry(QtCore.QRect(15, 260, 47, 13))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_3.setText("Name")
        # self.label_4 = QtWidgets.QLabel(parent=self.groupBox_3)
        # self.label_4.setGeometry(QtCore.QRect(233, 260, 91, 16))
        # self.label_4.setObjectName("label_4")
        # self.label_4.setText("Department")
        # self.departmentTextBox = QtWidgets.QLineEdit(parent=self.groupBox_3)
        # self.departmentTextBox.setGeometry(QtCore.QRect(230, 280, 141, 20))
        # self.departmentTextBox.setObjectName("departmentTextBox")
        self.emailAddressTextBox = QtWidgets.QLineEdit(parent=self.groupBox_3)
        self.emailAddressTextBox.setGeometry(QtCore.QRect(250, 280, 350, 20))
        self.emailAddressTextBox.setObjectName("emailAddressTextBox")
        self.label_5 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(250, 260, 101, 16))
        self.label_5.setObjectName("label_5")
        self.label_5.setText("Email Adress")
        self.sendImageButton = QtWidgets.QPushButton(parent=self.groupBox_3)
        self.sendImageButton.setGeometry(QtCore.QRect(620, 270, 101, 41))
        self.sendImageButton.setObjectName("sendImageButton")
        self.sendImageButton.setText("Send Photo")
        self.sendImageButton.clicked.connect(self.send_email)
        self.sendImageButton.setDisabled(True)

    def update(self):
        self.label.adjustSize()

    def browse_background_image(self):
        browse_file = QFileDialog.getOpenFileName(
            None, 'Choose Background', './Images/BG/', "Image files (*.jpg *.jpeg *.png *.gif)")[0]
        self.bg_img_path = browse_file
        if self.bg_img_path != '':
            print("Loading background image...")
            scene = QtWidgets.QGraphicsScene()
            pixmap = QPixmap(self.bg_img_path)
            pixmap = pixmap.scaledToWidth(270)  # resize the image canvas
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)

            self.bgImageGraphicsView.setScene(scene)
            # self.originalImageGraphicsView.resize(pixmap.width()+10, pixmap.height()+10)

            # Enable segment button
            self.bg_image_flag = 1
            self.show_mixed_bg_frame()
            print("Done!")

    def browse_frame(self):
        browse_file = QFileDialog.getOpenFileName(
            None, 'Choose Frame', './Images/Frame/', "Image files (*.png)")[0]
        self.frame_path = browse_file
        if self.frame_path != '':
            print("Loading frame...")
            scene = QtWidgets.QGraphicsScene()
            pixmap = QPixmap(self.frame_path)
            pixmap = pixmap.scaledToWidth(270)  # resize the image canvas
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)

            self.frameGraphicsView.setScene(scene)
            # self.originalImageGraphicsView.resize(pixmap.width()+10, pixmap.height()+10)

            # Enable segment button
            self.frame_flag = 1
            self.show_mixed_bg_frame()
            print("Done!")
    
        # print("loop exiting")
        # cam.release()
        # cv2.destroyAllWindows()
        # print("loop exiting successfully")
    def show_mixed_bg_frame(self):
        if self.bg_image_flag == 1 and self.frame_flag == 1:
            # Load the background image
            bg_img = Image.open(self.bg_img_path)
            # bg_img.show()

            # Load the frame image
            frame_img = Image.open(self.frame_path)
            # frame_img.show()
            # Ensure the foreground image has an alpha channel
            if 'A' not in frame_img.getbands():
                frame_img.putalpha(255)  # Add a fully opaque alpha channel

            # # Calculate the position to paste the foreground image at the center
            # position = ((bg_img.width - frame_img.width) // 2, (bg_img.height - frame_img.height) // 2)

            bg_img.paste(frame_img, (0, 0), frame_img)
            # bg_img.show("mixed image")

            bg_img.save("temp_bg.png")
            print("saved")

            scene = QtWidgets.QGraphicsScene()
            pixmap = QPixmap("temp_bg.png")
            pixmap = pixmap.scaledToWidth(270)  # resize the image canvas
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)

            self.outputImageGraphicsView.setScene(scene)
            self.captureImageButton.setDisabled(False)
            print("done")


    def capture_image(self):
        # cam = cv2.VideoCapture(2)
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # while cam.isOpened():
        cv2.namedWindow("Camera")
        print("Captured")

        while True:
            ret, frame = cam.read()

            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow("Camera", frame)

            k = cv2.waitKey(1)

            # if k % 256 == 27:  # ESC key to exit
            #     break
            # elif k % 256 == 32:  # Space key to capture image
            if k % 256 == 32:  # Space key to capture image
                img_name = "./bnet/click/clicked0.png"
                # img_name = "clicked0.png"

                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                cv2.destroyAllWindows()
                cam.release()

                # Save the original image
                original_filename = 'original/original_{}.png'.format(
                    self.img_counter)
                while os.path.exists(original_filename):
                    self.img_counter += 1
                    original_filename = 'original/original_{}.png'.format(
                        self.img_counter)

                img = Image.open(img_name)
                img.save(original_filename, format='png')


                input_dir  = "./bnet/click/"
                pred_dir = "./bnet/mask/"
                result_dir = "./bnet/result/"
                model_dir = "./saved_models/basnet_bsi/basnet.pth"

                process_images(input_dir, pred_dir, result_dir, model_dir)

                #  result = cv2.imread(result_path)

                only_masked_picture_but_transparent = os.path.join(result_dir, f'result0.png')


                bg_img = Image.open(self.bg_img_path)
                # print("bg_img size before resize:", bg_img.size)

                foreground_img = Image.open(only_masked_picture_but_transparent)
                bg_img = bg_img.resize((foreground_img.width, foreground_img.height))
                # print("bg_img size after resize:", bg_img.size)

                # foreground_img = Image.open(only_masked_picture_but_transparent)
                
                center_x = (bg_img.width - foreground_img.width) // 2
                center_y = (bg_img.height - foreground_img.height) // 2
                position = (center_x, center_y)
                result = Image.alpha_composite(bg_img, Image.new("RGBA", bg_img.size, (0, 0, 0, 0)))
                result.paste(foreground_img, position, foreground_img)


                result_path = './bnet/result/composite_result.png'
                result.save(result_path)
                # result.save(result_path)
                # result.save(result_path)
                # result.show()

                # result.show()

                # # Save or display the result
                frame = Image.open(self.frame_path)
                # result_path = './bnet/result/composite_result.png'
                open_oldBG = Image.open(result_path).convert("RGBA")    
                open_oldBG.paste(frame, (0, 0), frame)
                self.bg_result_path = './bnet/final_output/final{}.jpg'.format(self.img_counter)
                while os.path.exists(self.bg_result_path):
                    self.img_counter += 1
                    self.bg_result_path = './bnet/final_output/final{}.jpg'.format(self.img_counter)
                    

                bg_img_rgb = open_oldBG.convert('RGB')
                bg_img_rgb.save(self.bg_result_path, format='jpeg')
                # open_oldBG.show()
                cv2.destroyAllWindows()
                print("Process finished")
                self.img_counter += 1
                break
        scene = QtWidgets.QGraphicsScene()
        pixmap = QPixmap(self.bg_result_path)
        pixmap = pixmap.scaledToWidth(270)
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)

        self.outputImageGraphicsView.setScene(scene)

        result_img = Image.open(self.bg_result_path)
        result_img.show()
        
        cv2.waitKey(0)

        self.sendImageButton.setDisabled(False)
    
    def send_email(self):
        recipient_name = self.nameTextBox.text()
        recipient_email = self.emailAddressTextBox.text()
        # recipient_department = self.departmentTextBox.text()

        if recipient_name and recipient_email:
            smtp_server = "smtp.gmail.com"
            port = 587  # For starttls
            sender_email = mail.SENDER_EMAIL
            password = mail.PASSWORD
            

            # Create a multipart message
            message = MIMEMultipart()
            message['Subject'] = "Memories From University Day Project Showcase of CSE Department"
            message['From'] = sender_email
            message['To'] = recipient_email

            # Add text content to the message
            text_content = "Greetings from Visual Machine Intelligence Lab (VMI Lab). \nThank you for being a part of the project showcase of CSE Department from the University Day of JUST! We are as excited as you and hope you have enjoyed the event as much as we did. As a token of appreciation, here is a memorable picture from the event. This picture was captured using Virtual Photo Booth. Feel free to share it with your friends and relive the joyous moments.\n\nBest regards,\nVisual Machine Intelligence Lab (VMI Lab) \nand \nDepartment of Computer Science and Engineering\nJashore University of Science and Technology."

            text_part = MIMEText(text_content)
            message.attach(text_part)

            # Attach the image
            with open(self.bg_result_path, 'rb') as image_file:
                image_part = MIMEImage(
                    image_file.read(), name=f"{recipient_name.title()}.jpg")
                message.attach(image_part)

            # Create a secure SSL context
            context = smtplib.ssl.create_default_context()

            # Try to log in to the server and send the email
            try:
                server = smtplib.SMTP(smtp_server, port)
                server.ehlo()  # Can be omitted
                server.starttls(context=context)  # Secure the connection
                server.ehlo()  # Can be omitted
                server.login(sender_email, password)
                server.sendmail(
                    sender_email, message['To'], message.as_string())
                QMessageBox.about(
                    self, "Success", "Photo has been sent to your email. Thank you.")

                file_list = glob.glob('./bnet/final_output/*.jpg')
                file_list.sort(key=os.path.getmtime, reverse=True)
                if file_list:
                    # Get the path of the latest file
                    latest_file_path = file_list[0]
                    # Open and show the latest file
                    ## imgg = Image.open(latest_file_path)
                    save_directory = './latest_file/'
                    base_name = f"{recipient_name.title()}.jpeg"
                    save_path = os.path.join(save_directory, base_name)

                    counter = 0
                    while os.path.exists(save_path):
                        counter += 1
                        base_name = f"{recipient_name.title()}{counter}.jpeg"
                        save_path = os.path.join(save_directory, base_name)

                    imgg = Image.open(latest_file_path)
                    imgg.save(save_path, format='JPEG')

            except Exception as e:
                # Print any error messages to stdout
                print(e)
            finally:
                server.quit()
        else:
            QMessageBox.about(
                self, "Failed", "Please provide all the information")


def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec())
window()
 