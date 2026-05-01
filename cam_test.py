#!/usr/bin/env python3
from picamera2 import Picamera2, Preview
import time
import tflite_runtime.interpreter as tflite 

# picam2 = Picamera2()
# config = picam2.create_preview_configuration()
# picam2.configure(config)
# picam2.start()
# time.sleep(2)
# picam2.stop_preview()
# picam2.start_preview(True)
# time.sleep(2)

interpreter = tflite.Interpreter(model_path="/home/project/app/restnet50_cancer_classifier.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print(str(input_details[0]['shape']))
