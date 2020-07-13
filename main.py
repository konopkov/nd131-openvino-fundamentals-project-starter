"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client

def extract_objects(net_output, pt):
    objects_count = 0
    objects = []

    for obj in net_output:

        # obj[1] = 1 -> person
        # obj[2] -> probability
        if ((obj[1] == 1) and (obj[2] >= pt)):
            objects_count += 1
            objects.append({
                "id": obj[0],
                "label": obj[1],
                "probability": obj[2],
                "x_1": obj[3],
                "y_1": obj[4],
                "x_2": obj[5],
                "y_2": obj[6]
            })

    return objects, objects_count

def draw_boxes(frame, objects_dict):
    width = int(frame.shape[1]) 
    height = int(frame.shape[0])

    for obj in objects_dict:
        x_1 = int(obj["x_1"] * width)
        y_1 = int(obj["y_1"] * height)
        x_2 = int(obj["x_2"] * width)
        y_2 = int(obj["y_2"] * height)
        cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 0, 255), 1)

        obj_text = "prob: {0:.2g} ".format(obj["probability"])
        cv2.putText(frame, obj_text, org = (x_1, y_1 + 10), fontScale=0.3, fontFace = cv2.FONT_HERSHEY_SIMPLEX, color= (0, 0, 255))

    return frame

def draw_stats(frame, state):
    running_time_text = "Video length: {0:.4g} s".format(state["stats"]["video_length"])
    cv2.putText(frame, running_time_text, org = (10, 20), fontScale=0.5, fontFace = cv2.FONT_HERSHEY_SIMPLEX, color= (0, 0, 0))
    
    detection_current_duration_text = "Current duration: {0:.4g} s".format(state["detection_current"]["duration"])
    cv2.putText(frame, detection_current_duration_text, org = (10, 40), fontScale=0.5, fontFace = cv2.FONT_HERSHEY_SIMPLEX, color= (0, 0, 0))

    detection_total_count_text = "Current count: {0:.4g}".format(state["detection_current"]["count"])
    cv2.putText(frame, detection_total_count_text, org = (10, 60), fontScale=0.5, fontFace = cv2.FONT_HERSHEY_SIMPLEX, color= (0, 0, 0))

    detection_total_count_text = "Total count: {0:.4g}".format(state["detection_total"]["count"])
    cv2.putText(frame, detection_total_count_text, org = (10, 80), fontScale=0.5, fontFace = cv2.FONT_HERSHEY_SIMPLEX, color= (0, 0, 0))

    stats_average_infer_duration_text = "Inference time: {0:.4g} ms / {1:.4g} FPS".format(state["stats"]["average_infer_duration"] * 1000, 1.0 / state["stats"]["average_infer_duration"])
    cv2.putText(frame, stats_average_infer_duration_text, org = (10, 100), fontScale=0.5, fontFace = cv2.FONT_HERSHEY_SIMPLEX, color= (0, 0, 0))

    return frame

def skip_frames(current_count, remembered_count, current_skip_counter, max_skip_counter):
    if (current_count != remembered_count):
        if current_skip_counter < max_skip_counter:
            current_skip_counter += 1
            current_count = remembered_count
        else:
            current_skip_counter = 0

    return current_count, remembered_count, current_skip_counter, max_skip_counter

def update_state_with_stats(state, infer_start, running_time_start):
    state["stats"]["frames_counter"] += 1
    state["stats"]["accum_infer_time"] += time.time() - infer_start 
    state["stats"]["average_infer_duration"] = state["stats"]["accum_infer_time"] / state["stats"]["frames_counter"]
    state["stats"]["video_length"] = time.time() - running_time_start

    return state

def infer_on_stream(args, client, app_state): 
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :param app_state: Application state object
    :return: None
    """
    
    # Initials
    persons_prev_count = 0
    current_skip_counter = 0
    person_in_frame_start_time = 0
    infer_start = time.time()
    running_time_start = time.time()

    # Initialise the class
    infer_network = Network()

    ### Load the model through `infer_network` ###
    model = args.model
    infer_network.load_model(model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### Handle the input stream ###
    # Create a flag for single images
    image_flag = False
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True

    if (image_flag):
        max_skip_counter = 0
    else:    
        max_skip_counter = 5

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    ### Loop until stream is over ###
    while cap.isOpened():
        ### Read from the video capture ###
        flag, frame= cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]) )
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### Start asynchronous inference for specified request ###
        infer_start = time.time()
        infer_network.exec_net(p_frame)

        ### Wait for the result ###
        if infer_network.wait() == 0:

            ### Get the results of the inference request ###
            net_output = np.squeeze(infer_network.get_output())
            objects_in_frame, app_state["detection_current"]["count"] = extract_objects(net_output, args.prob_threshold)
            
            # Skip frames
            (app_state["detection_current"]["count"], 
            persons_prev_count, 
            current_skip_counter, 
            max_skip_counter) = skip_frames(
                app_state["detection_current"]["count"], 
                persons_prev_count, 
                current_skip_counter, 
                max_skip_counter)
            
            # Update stats
            app_state = update_state_with_stats(app_state, infer_start, running_time_start)

            # Draw bounding boxes
            frame = draw_boxes(frame, objects_in_frame)
            
            # Draw stats
            frame = draw_stats(frame, app_state)

            ### Extract any desired stats from the results ###
            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            # Person walked in
            if app_state["detection_current"]["count"] > persons_prev_count:
                person_in_frame_start_time = time.time()
                
                # Update state
                app_state["detection_total"]["count"] += (app_state["detection_current"]["count"] - persons_prev_count)
                persons_prev_count = app_state["detection_current"]["count"]

                # Publish count and total
                client.publish("person", json.dumps({
                        "count": app_state["detection_current"]["count"],
                        "total": app_state["detection_total"]["count"]
                    }))

            # Person walked out
            elif (app_state["detection_current"]["count"] < persons_prev_count):

                # Update state
                app_state["person_in_frame_duration"] = int(time.time() - person_in_frame_start_time)
                persons_prev_count = app_state["detection_current"]["count"]

                # Publish duration
                client.publish("person/duration", json.dumps({"duration": app_state["person_in_frame_duration"]}))
        
                # Publish count and total
                client.publish("person", json.dumps({
                        "count": app_state["detection_current"]["count"],
                        "total": app_state["detection_total"]["count"]
                    }))

            if (app_state["detection_current"]["count"] > 0): 
                app_state["detection_current"]["duration"]  = int(time.time() - person_in_frame_start_time)
            else:
                app_state["detection_current"]["duration"] = 0
            
        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
    
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Load the network and parse the output.

    :return: None
    """

    # Init appllication state
    app_state = {
        "person_in_frame_duration": 0,
        "detection_total": {
            "count": 0
        },
        "detection_current": {
            "count": 0,
            "duration": 0
        },
        "stats": {
            "accum_infer_time": 0,
            "average_infer_duration": 0,
            "video_length": 0,
            "frames_counter": 0
        }
    }

    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client, app_state)
    # Disconnect from MQTT server
    client.disconnect()

if __name__ == '__main__':
    main()
