import requests
import bs4
import pandas as pd
import time
import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import random
from time import ctime
import webbrowser
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

import cv2
import numpy as np
import glob


# Comment the following options and driver objects too

def Areny_speaks1(audio_string):
    tts = gTTS(text=audio_string, lang='en')
    r = random.randint(1, 1000)
    audio_file = 'audio-' + str(r) + '.mp3'
    tts.save(audio_file)
    playsound.playsound(audio_file)
    print(audio_string)
    os.remove(audio_file)


def ObjectDetectionImage():
    # Load Yolo
    net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading image
    # img = cv2.imread("room_ser.jpg")
    img = cv2.imread("street_1.jpg")

    # img = cv2.VideoCapture("Traffic.mp4")

    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
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

                cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        print(label)

        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    Voice()


def ObjectDetectionCam():
    # Load Yolo
    net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
    classes = []

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading image
    # img = cv2.imread("images/living_room.jpg")
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)

    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0
    while True:
        _, frame = cap.read()
        frame_id += 1

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

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

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    Voice()


def ObjectDetctionQubaCam():
    # Load Yolo
    net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

    # Name custom object
    classes = ["Quba Mosque"]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading image
    # img = cv2.imread("images/living_room.jpg")
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)

    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0
    while True:
        _, frame = cap.read()
        frame_id += 1

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

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

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                if label == 'Quba Mosque':
                    Areny_speaks1("The Quba Mosque (Masjid Qubā) — located in Medina, western Saudi Arabia."
                                  " It is the first mosque in Islamic history, and the oldest mosque in the world,"
                                  "originally completed in 622 CE. It was founded by the Islamic prophet Muhammad peace"
                                  " be upon him. Its first stones were positioned by Muhammad as soon as he arrived on his "
                                  "emigration from the city of Mecca to Medina and the mosque was completed by his companions.")

                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    Voice()


def ObjectDetctionQubaImg():
    # Load Yolo
    net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

    # Name custom object
    classes = ["Quba Mosque"]

    # Images path
    # images_path = glob.glob(r"C:\Users\UPM\Desktop\QubaMosque\*.jpg")

    images_path = glob.glob(r"C:\Drivers*.jpg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Insert here the path of your images
    random.shuffle(images_path)
    # loop through all the images
    for img_path in images_path:
        # Loading image
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=1.4, fy=1.4)
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
                if confidence > 0.3:
                    # Object detected
                    print(class_id)
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
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

        if label == 'Quba Mosque':
            Areny_speaks1("The Quba Mosque (Masjid Qubā) — located in Medina, western Saudi Arabia."
                          " It is the first mosque in Islamic history, and the oldest mosque in the world,"
                          "originally completed in 622 CE. It was founded by the Islamic prophet Muhammad peace"
                          " be upon him. Its first stones were positioned by Muhammad as soon as he arrived on his "
                          "emigration from the city of Mecca to Medina and the mosque was completed by his companions.")

        cv2.imshow("Image", img)
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    Voice()


reco = sr.Recognizer()
day = ctime()
day = day.split(' ')
day = int(day[3].split(':')[0])


def Gmap(need):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(executable_path='/usr/bin/chromedriver', options=options)
    driver.get('https://www.google.com/maps/@24.4384307,39.6172767,18z')
    time.sleep(2)
    driver.find_element_by_xpath('//*[@id="searchboxinput"]').click()
    driver.find_element_by_xpath('//*[@id="searchboxinput"]').send_keys(need, Keys.ENTER)
    time.sleep(3)
    names = []
    url = driver.current_url
    response = requests.get(url)
    response = response.text
    data = bs4.BeautifulSoup(response, 'lxml')
    details = str(data)
    details = details.split("n,null,null,null,null,[null,\\")
    print(len(details))
    details = details[1:]
    print(len(details))
    for i in details:
        names.append(i.split("(المالك)")[0].replace('"', '')[:-1])
        print(i.split("(المالك)")[0].replace('"', '')[:-1])
    driver.close()
    return names


def Voice():
    def record_audio(ask=False):
        with sr.Microphone() as source:
            if ask:
                Areny_speaks(ask)
            audio = reco.listen(source)
            voice_data = ''
            try:
                voice_data = reco.recognize_google(audio)
                # Areny_speaks(voice_data)
            except sr.UnknownValueError:
                Areny_speaks('Sorry, I did not understand that')
            except sr.RequestError:
                Areny_speaks('Sorry, the server is not working')

            return voice_data

    def responed(voice_data):
        if 'hello' in voice_data:
            Areny_speaks('hello')
        elif 'thank you' in voice_data:
            Areny_speaks('you are welcome')
            exit()
        elif "your name" in voice_data:
            Areny_speaks('my name is Areny')
        elif 'what is the time' in voice_data:
            Areny_speaks(ctime())
        elif 'how are you' in voice_data:
            Areny_speaks('I am good, thank you')
        elif 'exit' in voice_data:
            Areny_speaks('exiting now,.. have a nice day')
            exit()
        elif 'tell me' in voice_data:
            information = record_audio('What do you want to know about')
            url = 'https://google.com/search?q=' + information
            webbrowser.get().open(url)
            Areny_speaks('Here is the information you requested about ' + information)
        elif 'find a location' in voice_data:
            location = record_audio('Where do you want to go?')
            url = 'https://google.nl/maps/place/' + location + '/@24.4382084,39.6336291,14z/data=!3m1!4b1'
            webbrowser.get().open(url)
            Areny_speaks('Here is the location of' + location)

        elif 'what is this' in voice_data:
            ObjectDetctionQubaCam()
        elif 'what building' in voice_data:
            ObjectDetectionCam()
        elif 'what am I seeing' in voice_data:
            ObjectDetectionImage()

        elif 'hotels nearby' in voice_data:
            names = Gmap("فندق")
            Answer = record_audio('what is the name of the hotel you want to book')
            print(Answer)
            for i in names:
                if (Answer).lower() in i.lower():
                    Areny_speaks("here's a link to booking")
                    url = 'https://www.agoda.com/' + i.replace(' ',
                                                               '-') + '/hotel/medina-sa.html?finalPriceView=0&isShowMobileAppPrice=false&cid=1587510&tag=hid5830760,pidX0JasgokJH4AA-wW1QUAAABf&numberOfBedrooms=&familyMode=false&isAgMse=false&ccallout=false&defdate=false&adults=2&children=0&rooms=1&maxRooms=9&checkIn=2020-12-31&childAges=&defaultChildAge=8&travellerType=-1&tspTypes=6&los=1&searchrequestid=12cb90a8-2425-4089-b434-f382daa128f9'
                    webbrowser.get().open(url)
                    break

        elif 'restaurants nearby' in voice_data:
            Gmap('مطعم')
            Areny_speaks("Here's a list of nearby restaurants")

        elif 'best hotels' in voice_data:

            options = Options()
            options.headless = True
            driver = webdriver.Chrome('D:\Program Files\chromedriver.exe', options=options)
            driver.get('https://www.tripadvisor.com/Tourism-g298551-Medina_Al_Madinah_Province-Vacations.html')
            driver.find_element_by_xpath('//*[@id="lithium-root"]/main/div[2]/div/div/div[1]/a').click()
            url = driver.current_url
            response = requests.get(url)
            response = response.text
            data = bs4.BeautifulSoup(response, 'lxml')
            names = data.select('.listing_title')
            prices = data.select('.price-wrap')
            name = []
            for i in range(len(names)):
                x = names[i].text
                name.append(x)

            price = []
            for i in prices:
                x = i.text
                x = x.replace("\xa0", "")
                x = x.lstrip()
                x = x.split(" ")
                o = x[0].count('SAR')
                if o > 1:
                    x = x[0].split('SAR')
                    price.append('SAR' + x[-1])
                else:
                    if (len(x) > 1):
                        price.append(str(x[1]))
                    else:
                        price.append(str(x[0]))

            form = pd.DataFrame(price, name)

            Areny_speaks("here's a list of best hotels and the price of one night")
            print(form)
            driver.close()
        elif 'places to eat' in voice_data:

            driver = webdriver.Chrome("D:\Program Files\chromedriver.exe")
            time.sleep(2)
            driver.get('https://www.tripadvisor.com/Tourism-g298551-Medina_Al_Madinah_Province-Vacations.html')
            driver.find_element_by_xpath('//*[@id="lithium-root"]/main/div[2]/div/div/div[4]/a').click()
            time.sleep(8)

            ##################
            if (day > 4) and (day <= 10):
                meal = ' 1'
            elif (day > 10) and (day <= 12):
                meal = '2'
            elif (day > 12) and (day <= 18):
                meal = '3'
            else:
                meal = '4'
            ##################

            driver.find_element_by_xpath(
                '//*[@id="component_47"]/div/div[3]/div[2]/div[' + meal + ']/div/label').click()
            url = driver.current_url
            response = requests.get(url)
            response = response.text
            data = bs4.BeautifulSoup(response, 'lxml')
            a_tags = data.find_all("a", attrs={"class": "_15_ydu6b"})
            for i in range(len(a_tags)):
                x = a_tags[i]
                x = str(x)
                x = x.split('<!-- -->. <!-- -->')[1].replace("</a>", "")

                print(x)
            Areny_speaks("here's a list of best restaurants in the city")
            driver.close()
        else:
            Areny_speaks("Did you say " + voice_data + " ?")

    def Areny_speaks(audio_string):
        tts = gTTS(text=audio_string, lang='en')
        r = random.randint(1, 1000)
        audio_file = 'audio-' + str(r) + '.mp3'
        tts.save(audio_file)
        playsound.playsound(audio_file)
        print(audio_string)
        os.remove(audio_file)

    time.sleep(1)
    Areny_speaks('How May I help?')
    while 1:
        voice_data = record_audio()
        responed(voice_data)


def main():
    Voice()


if __name__ == "__main__":
    main()
