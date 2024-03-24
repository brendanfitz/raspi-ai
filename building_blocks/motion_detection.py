from gpiozero import MotionSensor
import time

pir = MotionSensor(18)

while True:
    pir.wait_for_motion()
    print("You moved!")
    pir.wait_for_no_motion()
    time.sleep(0.01)
