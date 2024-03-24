import pygame

pygame.init()
fart = pygame.mixer.Sound('fart.mp3')
playing = fart.play()

while playing.get_busy():
    pygame.time.delay(100)