from typing import List

import numpy as np
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.prediction import PredictionResult
from utils import create_arg_parser, obtain_detection_model

from norfair import Detection, Tracker, Video, draw_boxes, draw_tracked_boxes
from norfair.filter import OptimizedKalmanFilterFactory
import cv2
import numba as nb
import math
from pyfirmata import Arduino, util
from serial.tools import list_ports
import traceback
import time
import socket
from imutils.video import VideoStream
import imagezmq
from webservice import db, TrackedObjects
import uuid
# Numba Python -> C Translator (+25 - 35% performance increase on CPU and +50% Inference Speed on GPU)
@nb.jit(nopython=True)
def point_side_of_line(point, line):
    """
    Determine which side of a vertical line a point is on.

    Args:
        point: A tuple (x, y) representing the point to test.
        line: A tuple ((x1, y1), (x2, y2)) representing the line with a 90 degree slope.

    Returns:
        A string indicating which side of the line the point is on. Possible values are 'left', 'right', or 'on'.
    """
    # Convert the point and line tuples to NumPy arrays
    point_arr = np.array(point)
    line_arr = np.array(line)
    # Extract the x-coordinate of the line
    x = line_arr[0, 0]
    # Calculate the sign of the difference between the x-coordinate of the point and the x-coordinate of the line
    side = math.copysign(1, point_arr[0] - x)
    # Return the appropriate string based on the sign
    if side == -1:
        return 'L'
    elif side == 1:
        return 'R'
    else:
        return 'O'


class ShipTracker:
    def __init__(self, DEBUG=True):
        print("[+] Starting the Ship Tracking System Version 1.0")
        self.DEBUG = DEBUG
        # define the parameters for YOLOv5 Model and Norfair Tracker
        self.define_parameters()
        # Obtain the Detection Model
        self.detection_model = obtain_detection_model(self.model_confidence_threshold)
        # Initialize the Norfair Tracker using the Defined Parameters
        self.tracker = Tracker(
            initialization_delay=self.initalization_delay,
            distance_function="iou",
            hit_counter_max=self.hit_counter_max,
            filter_factory=OptimizedKalmanFilterFactory(),
            distance_threshold=self.distance_threshold,
        )
        # reading the video input stream from OpenCV
        self.video = cv2.VideoCapture("http://localhost:8080")
        # creating a numpy based python dict for faster performance, and low memory usage
        self.temp_db = {
            "tracking_id":[],
            "current_location":[],
            "last_pulse":[],
        }
        # Finding Connected Arduino Boards
        serial_response = self.com_ports()
        if serial_response == False:
            print("[!] No Arduino Boards Found - Exiting Peacefully")
            print("[+] Try Reconnecting the Arduino Board and Restart the Program")
            exit(0)
        # Connecting the the Arduino Uno Board
        self.establish_connection()
        # Initializing the Arduino Board Pins
        self.declare_pulse_variable()
        # Starting the ImaeZMQ Server
        self.setup_connection()
        self.test_configuration()



        while True:
            # getting the live video frame from OpenCV
            _, self.frame = self.video.read()
            # resizing the frame to 640x480 to go ahead and get the faster prediction
            self.frame = cv2.resize(self.frame, (640, 480))
            # getting the prediction from the detection model
            self.result = get_sliced_prediction(self.frame, self.detection_model) # this will automatically determine the best size for prediction slices
            # Drawing Lines and Circles for End-User Refrence
            self.circle_and_lines()
            # getting the detections from the prediction result
            self.detections = self.get_detections(self.result.object_prediction_list)
            # updating the tracker with the detections
            self.tracked_objects = self.tracker.update(detections=self.detections)
            for tracked_object in self.tracked_objects:
                if tracked_object.hit_counter > 10:
                    self.last_detection_points = tracked_object.last_detection.points
                    # putting a circle on the last detection point
                    cv2.circle(self.frame, (int(self.last_detection_points[0][0]), int(self.last_detection_points[0][1])), 5, (0, 255, 0), -1)
                    self.current_stern_location = (int(self.last_detection_points[0][0]), int(self.last_detection_points[0][1]))
                    # now finding the relative location of the stern
                    self.relative_location_center = point_side_of_line(self.current_stern_location, self.center_line)
                    self.relative_location_left = point_side_of_line(self.current_stern_location, self.left_line)
                    self.relative_location_right = point_side_of_line(self.current_stern_location, self.right_line)
                    # now printing the relative location of the stern on the OpenCV Frame
                    cv2.putText(self.frame, f"Center: {self.relative_location_center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 120), 2)
                    cv2.putText(self.frame, f"Left: {self.relative_location_left}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 120), 2)
                    cv2.putText(self.frame, f"Right: {self.relative_location_right}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 120), 2)
                    if tracked_object.id not in self.temp_db["tracking_id"]:
                        self.temp_db["tracking_id"].append(tracked_object.id)
                        # putting the most relative position in the database
                        """
                        This takes into account the best course of action for existing ships that might be included in the frame.
                        It also sends all of the signals for each one of the ships in a systematic manner provided we just started detection, and
                        it has already crossed the determined  reigons.
                        """
                        if self.relative_location_left == "L":
                            self.temp_db["current_location"].append("L-Left")
                            # send the start tracking signal here
                            self.send_signal("st")
                        if self.relative_location_left == "R" and self.relative_location_right == "L":
                            self.temp_db["current_location"].append("R-Left")
                            # send the start tracking signal here
                            self.send_signal("st")
                            # then send the crossed left signal here
                            self.send_signal("cl")
                        if self.relative_location_center == "R" and self.relative_location_right == "L":
                            self.temp_db["current_location"].append("R-Center")
                            # send the start tracking signal here
                            self.send_signal("st")
                            # then send the crossed left signal here
                            self.send_signal("cl")
                            # then send the crossed center signal here
                            self.send_signal("cc")
                        if self.relative_location_center == "R" and self.relative_location_right == "R":
                            self.temp_db["current_location"].append("R-Right")
                            # send the start tracking signal here
                            self.send_signal("st")
                            # then send the crossed left signal here
                            self.send_signal("cl")
                            # then send the crossed center signal here
                            self.send_signal("cc")
                            # then send the crossed right signal here
                            self.send_signal("cr")
                            self.send_signal("et")
                        
                    if tracked_object.id in self.temp_db["tracking_id"]:
                        # if we already have the id of the tracked object over here.
                        # then we need to check if the current location is the same as the last location
                        # if it is the same, then we don't need to do anything
                        # if it is different, then we need to send the signal to the arduino board
                        # and then update the database
                        # getting the index of the tracked object
                        self.index_n = self.temp_db["tracking_id"].index(tracked_object.id)
                        # getting the current location of the tracked object
                        self.current_location_n = self.temp_db["current_location"][self.index_n]
                        # now checking if the current location is the same as the last location
                        if self.current_location_n == "L-Left":
                            if self.relative_location_left == "L":
                                print(f"[+] No New Signals to Send for Object: {tracked_object.id}")
                                continue
                        if self.current_location_n == "R-Left":
                            if self.relative_location_center == "L":
                                if self.relative_location_left == "R":
                                    print(f"[+] No New Signals to Send for Object: {tracked_object.id}")
                                    continue
                        if self.current_location_n == "R-Center":
                            if self.relative_location_center == "R":
                                if self.relative_location_right == "L":
                                    print(f"[+] No New Signals to Send for Object: {tracked_object.id}")
                                    continue
                        if self.current_location_n == "R-Right":
                            if self.relative_location_right == "R":
                                print(f"[+] No New Signals to Send for Object: {tracked_object.id}")
                                continue
                        print(f"[+] Object Tracking Still Continued for Object No. {tracked_object.id}")
                        if self.relative_location_left == "L":
                            if self.current_location_n != "L-Left":
                                # getting the current location here
                                if self.relative_location_center == "L":
                                    self.temp_db["current_location"][self.index_n] = "R-Left"
                                    # send the start tracking signal here
                                    self.send_signal("cl")
                                if self.relative_location_center == "R" and self.relative_location_right == "L":
                                    self.temp_db["current_location"][self.index_n] = "R-Center"
                                    # send the start tracking signal here
                                    self.send_signal("st")
                                    # then send the crossed left signal here
                                    self.send_signal("cl")
                                if self.relative_location_right == "R":
                                    self.temp_db["current_location"][self.index_n] = "R-Right"
                                    # send the ship has crossed signal over here.
                                    self.send_signal("cr")
                                    self.send_signal("et")
                        if self.relative_location_left == "R" and self.relative_location_right == "L":
                            if self.current_location_n != "R-Left":
                                # set the current location as R-Left
                                self.temp_db["current_location"][self.index_n] = "R-Left"
                                # send the crossed left signal here
                                self.send_signal("cl")
                        if self.relative_location_center == "R" and self.relative_location_right == "L":
                            if self.current_location_n != "R-Center":
                                # set the current location as R-Center
                                self.temp_db["current_location"][self.index_n] = "R-Center"
                                # send the crossed center signal here
                                self.send_signal("cc")
                        if self.relative_location_right == "R":
                            if self.current_location_n != "R-Right":
                                # set the current location as R-Right
                                self.temp_db["current_location"][self.index_n] = "R-Right"
                                # send the crossed right signal here
                                self.send_signal("cr")
                                self.send_signal("et")
                        
                    
            # drawing the tracked boxes on the frame
            self.frame = draw_tracked_boxes(self.frame, self.tracked_objects)
            # showing the frame
            _, self.frame_bytes = cv2.imencode(".jpg", self.frame)
            with open('frame.jpg', 'wb') as frame_writer:
                frame_writer.write(self.frame_bytes.tobytes())
            cv2.imshow("frame", self.frame)
            # if the user presses the "q" key, then break the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    def declare_pulse_variable(self):
        """
        Connection Parameters to Control the Arduino Board
        Copied AS-IS from https://github.com/diogoryu/tracking to Ensure Compatibility
        """
        self.alarm = self.arduino.get_pin('d:10:o')  # Buzzer
        self.led_power_ok = self.arduino.get_pin('d:9:o')  # Led power
        self.man_auto = self.arduino.get_pin('d:8:o')  # Acionamento manual ou automático # automatic or manual
        self.saida1 = self.arduino.get_pin('d:7:o')  # Inicio de medição - begin of tracking
        self.saida2 = self.arduino.get_pin('d:6:o')  # Top proa
        self.saida3 = self.arduino.get_pin('d:5:o')  # Top popa
        self.saida4 = self.arduino.get_pin('d:4:o')  # Fim de Medição end of tracking
        self.saida5 = self.arduino.get_pin('d:3:o')  # Relé erro

        # Portas analógicas
        self.power_ok = self.arduino.get_pin('a:0:i')  # Sinal analógico da fonte
        self.feedback1 = self.arduino.get_pin('a:1:i')  # Feedback da saída 1
        self.feedback2 = self.arduino.get_pin('a:2:i')  # Feedback da saída 2
        self.feedback3 = self.arduino.get_pin('a:3:i')  # Feedback da saída 3
        self.feedback4 = self.arduino.get_pin('a:4:i')  # Feedback da saída 4

        self.state_man_auto = 1  # variável de controle do botão manual/automático


        return None
    def load_configuration(self):
        self.man_auto.write(self.state_man_auto)
        self.initial_declaration = 0
        self.saida5.write(0)
        self.alarm.write(0)
        self.power = self.power_ok.read()
        self.feed1 = self.feedback1.read()
        self.feed2 = self.feedback2.read()
        self.feed3 = self.feedback3.read()
        self.feed4 = self.feedback4.read()
        while self.power <= 0.900:
            self.led_power_ok.write(1)
            self.alarm.write(1)
            self.arduino.digital[3].write(1)
            time.sleep(0.2)
            self.led_power_ok.write(0)
            time.sleep(0.2)
            self.power = self.power_ok.read()
            # ignoring the flag logic here, unsure what it does
        # loading the configuration file
        with open("configuration.json", "r") as config_reader:
            self.config = eval(config_reader.read())
        # setting the manual / automatic configuration
        if self.config["manauto"] == True:
            self.state_man_auto = 1
            self.man_auto.write(self.state_man_auto)
        if self.config["manauto"] == False:
            self.state_man_auto = 0
            self.man_auto.write(self.state_man_auto)
        return True
    def test_configuration(self):
        """
        This Function Tests to Connected Equipment to Arduino And Relays Back the Failed Connections.
        It has the Capability to Detect which Equipment has Failed and Which Equipment has not failed.
        """
        print("[+] Testing the Arduino Configuration, Please Wait ....")
        errors = []
        error_dict = {
            'errors':errors
        }
        cont = 0
        self.saida1.write(1);
        time.sleep(0.25)
        feed1 = self.feedback1.read()
        try:
            if feed1 >= 1.0:
                cont = cont + 1
        except:
            if cont != 1:
                print("[+] Saida1 Pin and Feedback1 Pin have Issues")
                errors.append("ERROR: CHECK PIN a:1:i (Side1)")
                cont = 1
        self.saida1.write(0)


        self.saida2.write(1)
        time.sleep(0.25)
        feed2 = self.feedback2.read()
        try:
            if feed2 >= 1.0:
                cont = cont + 1
        except:
            if cont != 2:
                print("[+] Said2 Pin and Feedback2 Pin have Issues")
                errors.append("ERROR: CHECK PIN a:2:i (Side2)")
                cont = 2
        
        self.saida2.write(0)
        

        self.saida3.write(1);
        time.sleep(0.25)
        feed3 = self.feedback3.read()
        try:
            if feed3 >= 1.0:
                cont = cont + 1
        except:
            if cont != 3:
                print("[+] Saida3 Pin and Feedback3 Pin have Issues")
                errors.append("ERROR: CHECK PIN a:3:i (Side3)")
                cont = 3
        self.saida3.write(0)
        
        self.saida4.write(1)
        time.sleep(0.25)
        feed4 = self.feedback4.read()
        try:
            if feed4 >= 1.0:
                cont = cont + 1
        except:
            if cont != 4:
                print("[+] Saida4 Pin and Feedback4 Pin have Issues")
                errors.append("ERROR: CHECK PIN a:4:i (Side4)")
                cont = 4

        self.saida4.write(0)
        self.saida5.write(1)
        self.alarm.write(1)
        self.saida5.write(0)
        self.alarm.write(0)
        with open('errors.json', 'w+') as error_writer:
            error_writer.write(str(error_dict))

    def load_config(self):
        with open("configuration.json", "r") as config_reader:
            self.config = eval(config_reader.read())
        self.reiniciar = self.config["reiniciar"]
        return self.config
    
    def send_signal(self, signal):
        """
        Args: Signal
        This Function is responsible for sending the Signals to the Arduino Board.
        There are 5 Signal Types:
        1. 'st': Start Tracking
        2. 'cl': Crossed Left
        3. 'cc': Crossed Center
        4. 'cr': Crossed Right
        5. 'et': End Tracking
        Example Usage:
        self.send_signal('st') -> Sends the Start Tracking Pulse using Arduino Uno.
        Returns: True if the Signal was Sent Successfully.
        Returns: False if the Signal was not Sent Successfully. (Connection Error)
        """
        self.load_config()
        # everytime a signal is sent we will add the information to the database
        randomized_image = f"{uuid.uuid4()}.jpg"
        cv2.imwrite(f"./images/{randomized_image}", self.frame)
        new_object = TrackedObjects(ship_image=randomized_image, last_known_location=signal)
        db.session.add(new_object)
        db.session.commit()
        with open("temp.json", "w+") as temp_writer:
            temp_writer.write(str(self.temp_db))
        if signal == 'st':
            # Start Tracking Signal (Inicio de medição)
            self.saida1.write(1)
            self.feed1 = self.feedback1.read()
            print(f"[+] Arduino Inicio Feedback: {self.feed1}")
            if self.feed1 == None:
                if self.DEBUG == True:
                    self.feed1 = 0.8
            while self.feed1 < 1.0 and self.reiniciar == 0:
                self.led_power_ok.write(1)
                self.alarm.write(1)
                self.arduino.digital[3].write(1)
                self.alarm.write(0)
                self.led_power_ok.write(0)
                self.load_config()
                if self.reiniciar == 1:
                    break
            self.saida1.write(0)
        if signal == "cl":
            self.saida2.write(1)
            self.feed2 = self.feedback2.read()
            if self.feed2 == None:
                if self.DEBUG == True:
                    self.feed2 = 0.8
            print(f"[+] Arduino Top Proa Feedback: {self.feed2}")
            while self.feed2 < 1.0 and self.reiniciar == 0:
                self.led_power_ok.write(1)
                self.alarm.write(1)
                self.arduino.digital[3].write(1)
                self.alarm.write(0)
                self.led_power_ok.write(0)
                self.load_config()
                if self.reiniciar == 1:
                    break
            self.saida2.write(0)
        if signal == "cc":
            self.saida3.write(1)
            self.feed3 = self.feedback3.read()
            print(f"[+] Arduino Top Popa Feedback: {self.feed3}")
            if self.feed3 == None:
                if self.DEBUG == True:
                    self.feed3 = 0.8
            while self.feed3 < 1.0 and self.reiniciar == 0:
                self.led_power_ok.write(1)
                self.alarm.write(1)
                self.arduino.digital[3].write(1)
                self.alarm.write(0)
                self.led_power_ok.write(0)
                self.load_config()
                if self.reiniciar == 1:
                    break
            self.saida3.write(0)
        if signal == "cr":
            self.saida4.write(1)
            self.feed4 = self.feedback4.read()
            if self.feed4 == None:
                if self.DEBUG == True:
                    self.feed4 = 0.8
            print(f"[+] Arduino Bottom Feedback: {self.feed4}")
            while self.feed4 < 1.0 and self.reiniciar == 0:
                self.led_power_ok.write(1)
                self.alarm.write(1)
                self.arduino.digital[3].write(1)
                self.alarm.write(0)
                self.led_power_ok.write(0)
                self.load_config()
                if self.reiniciar == 1:
                    break
            self.saida4.write(0)
        if signal == "et":
            self.saida5.write(1)
            while self.reiniciar == 0:
                self.led_power_ok.write(1)
                self.alarm.write(1)
                self.arduino.digital[3].write(1)
                self.alarm.write(0)
                self.led_power_ok.write(0)
                self.load_config()
                if self.reiniciar == 1:
                    break
            self.saida5.write(0)
        return True

            


    def com_ports(self):
        "Using PySerial to Get the Serial Ports and Connecting to the Arduino Uno Attached."
        ports = list_ports.comports()
        for port in ports:
            if "Arduino" in port.description:
                self.arduino_port = port.device
                return True # If Found Return True
        return False # Else Show that Not Found
    def establish_connection(self):
        try:
            self.arduino = Arduino(self.arduino_port, timeout=0.2) # Establishing the Connection
        except:
            print("[+] Failed Connection to the Arduino Uno. Make Sure You have Flashed it with Firmata and that it is Connected to the Computer.")
            print('[+] ------------------ Start Traceback ------------------')
            traceback.print_exc()
            print('[+] ------------------ End Traceback --------------------')
            exit(-1)
        
    
    def circle_and_lines(self):
        """
        This Function goes ahead and then generates Circles and Lines Representing Each one of the Point in Question.
        """
        # calculate the cx and cy of the frame
        self.cx = self.frame.shape[1] // 2
        self.cy = self.frame.shape[0] // 2
        # draw the center of the frame
        self.frame = cv2.circle(self.frame, (self.cx, self.cy), 5, (0, 255, 0), -1)
        # finding the vertical_line coordinates
        self.center_line = self.find_vertical_line(self.cx, self.cy)
        # draw the center line
        self.frame = cv2.line(
            self.frame, self.center_line[0], self.center_line[1], (0, 255, 0), 2
        )
        # draw at the left center of the frame
        self.frame = cv2.circle(self.frame, (40, self.cy), 5, (0, 255, 0), -1)
        # calculating the vertical line coordinates
        self.left_line = self.find_vertical_line(40, self.cy)
        # draw the left line
        self.frame = cv2.line(
            self.frame, self.left_line[0], self.left_line[1], (0, 255, 0), 2
        )
        # finding the co-ordinates of the right line
        self.right_line = self.find_vertical_line(self.cx + 240, self.cy)
        # draw the right line
        self.frame = cv2.line(
            self.frame, self.right_line[0], self.right_line[1], (0, 255, 0), 2
        )
        # draw at the right center of the frame
        self.frame = cv2.circle(self.frame, (self.cx + 240, self.cy), 5, (0, 255, 0), -1)
        return self.frame

    def find_vertical_line(self, x, y, img_width=640, img_height=480):
        """
        Function Optimized for Faster Performance on ShipTracking
        """
        # Calculate the y-coordinates of the intersection points
        y1, y2 = 0, img_height
        # Return the line coordinates as a tuple of two points
        return ((int(x), int(y1)), (int(x), int(y2)))
    def define_parameters(self):
        self.model_confidence_threshold = 0.6
        self.initalization_delay = 3
        self.hit_counter_max = 500
        self.distance_threshold = 0.3
    def get_detections(self, object_prediction_list: PredictionResult) -> List[Detection]:
        detections = []
        for prediction in object_prediction_list:
            if prediction.category.id == 8:
                bbox = prediction.bbox

                detection_as_xyxy = bbox.to_voc_bbox()
                bbox = np.array(
                    [
                        [detection_as_xyxy[0], detection_as_xyxy[1]],
                        [detection_as_xyxy[2], detection_as_xyxy[3]],
                    ]
                )
                detections.append(
                    Detection(
                        points=bbox,
                        scores=np.array([prediction.score.value for _ in bbox]),
                        label=prediction.category.id,
                    )
                )
        return detections
    def setup_connection(self):
        self.rpi_name = socket.gethostname()
        # this method is depreciated
        #self.sender = imagezmq.ImageSender(connect_to='tcp://0.0.0.0:5000', REQ_REP=False)





if __name__ == "__main__":
    ShipTracker()