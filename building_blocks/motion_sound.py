from gpiozero import MotionSensor
import time
import pygame

pygame.init()
fart = pygame.mixer.Sound('fart.mp3')

pir = MotionSensor(18)

while True:
    pir.wait_for_motion()
    print("You moved!")
    playing = fart.play()
    while playing.get_busy():
        pygame.time.delay(100)
    pir.wait_for_no_motion()
