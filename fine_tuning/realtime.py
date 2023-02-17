# Untitled - By: phuon - Wed Dec 7 2022

import sensor, image, time, tf

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((96, 96))
sensor.skip_frames(time = 2000)

net = tf.load("vww_96_int8.tflite", load_to_fb=True)
labels = ['non-person', 'person']


clock = time.clock()

while(True):
    img = sensor.snapshot()
    classification_result = net.classify(img)
    model_output = classification_result[0].output()

    print(model_output)
