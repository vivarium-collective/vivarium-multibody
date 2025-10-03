## This file is an exploration to the pymunk and pygame engines to explore how to implement in biofilm formation
    #So far... box (cell) that goes to surface (cell surface)
    #TODO: pygame spings and cell forces (biologically relevant)

#TODO 10/01/2025: Change box to circle, add pivot joint, add damped spring

import pymunk
from pymunk import Space, pygame_util, Poly, Body, moment_for_box, Circle
import sys
import pygame

def main():

    #Initialize the Pygame Module and screen (where we draw the simulation result)
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    print_options = pygame_util.DrawOptions(screen)

    #Space Object - global variable and assign it a gravity vector
    space = Space()
    space.gravity = 0, 981 #confirm units

    #Segment - Fixed ground for our object
    surface = pymunk.Segment(space.static_body, (0, 500), (800, 500), 10)
    surface.elasticity = 1.0
    surface.friction = 1.0

    #Dynamic body with mass, moment and position
    body = Body(mass=10, moment=10)
    body.position = 100, 100

    #Create Circle shape and attach it to the body
    circle = Circle(body, radius=20.0)
    circle.elasticity = 5

    #Add body, circle, and surface to the space
    space.add(body, circle, surface)


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
        #shape = pymunk.Circle(body, radius=10)
        screen.fill((255, 255, 255))
        space.step(0.0001)
        space.debug_draw(print_options)
        pygame.display.flip()






if __name__ == '__main__':
    sys.exit(main())
