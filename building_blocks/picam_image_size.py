import picamera2
import time


with picamera2.Picamera2() as camera:
    # capture_config = camera.create_still_configuration()
    camera.resolution = (640, 480)
    camera.framerate = 30
    camera.annotate_text_size = 20
    # width, height, channels = common.input_image_size(interpreter)
    camera.start(show_preview=True)
    time.sleep(4)

    # array = camera.switch_mode_and_capture_array(capture_config, "main")
    array = camera.capture_array("main")
    print(array.shape)
    camera.stop()
# camera = picamera2.Picamera2()
# camera.resolution = (640, 480)
# camera.framerate = 30
# camera.annotate_text_size = 20
# camera.start_preview()
# camera.start()
# time.sleep(1)
# array = camera.capture_array("main")
# print(array.shape)

# camera.stop_preview()
# camera.stop()