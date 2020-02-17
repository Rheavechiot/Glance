# import kivy module
import os

import kivy
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
# base Class of your App inherits from the App class.
# app:always refers to the instance of your application
from gtts import gTTS
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
# From graphics module we are importing
# Rectangle and Color as they are
# basic building of canvas.
from kivy.graphics import Rectangle, Color

# The Label widget is for rendering text.
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget
import time

from playsound import playsound



class About(Screen):
    pass

class WindowManager(ScreenManager):
    pass

class Home(Screen):
    pass
    def image_detect(self):
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Loading image
        Tk().withdraw()
        filename = askopenfilename()
        if filename!="":
            img = cv2.imread(filename)
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            font = cv2.FONT_HERSHEY_PLAIN
            labels=""
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    labels+=" a "+label + " and"
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

            labels='I see '+' '.join(labels.split(' ')[:-1])
            cv2.imshow("Image", img)
            language = 'en-us'

            myobj = gTTS(text=labels, lang=language, slow=False)
            myobj.save("sound.mp3")
            playsound("sound.mp3")
            os.remove("sound.mp3")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    def live_detection(self):
        net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Loading image
        cap = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id = 0
        while True:
            _, frame = cap.read()
            frame_id += 1

            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (600, 600), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing information on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
            labels = ""
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    labels += " a "+label + " and "
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)

            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
            cv2.imshow("Image", frame)

            key = cv2.waitKey(1)

            language = 'en-us'
            if labels != "":
                labels = 'I see ' + ' '.join(labels.split(' ')[:-1])
                myobj = gTTS(text=labels, lang=language, slow=True)
                myobj.save("sound.mp3")
                playsound("sound.mp3")
                os.remove("sound.mp3")
            if key == 27:
                break



        cap.release()
        cv2.destroyAllWindows()

# Create the App Class
class GlanceApp(App):
    def build(self):
        return

    # run the App

if __name__=="__main__":
    GlanceApp().run()
