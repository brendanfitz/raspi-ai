from picamera2 import Picamera2, Preview
import time
picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start(show_preview=True)


time.sleep(5)
picam2.stop()