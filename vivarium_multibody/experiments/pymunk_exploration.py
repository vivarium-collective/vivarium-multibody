## This file is an exploration to the pymunk and pygame engines to explore how to implement in biofilm formation
    #So far... box (cell) that goes to surface (cell surface)
    #TODO: pygame spings and cell forces (biologically relevant)


import pymunk
from pymunk import Space, pygame_util, Poly, Body, moment_for_box
import sys
import pygame

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    space = Space()
    space.gravity = 0, 981
    #gravitational force (check units)
    #need to but these specifications into a for loop so we hava many being generated


    body = Body()
    body.position = 300, 300
    size = (50,50)
    poly = Poly.create_box(body, size=size)
    body.mass = 10
    body.moment = moment_for_box(body.mass,size)
    space.add(body, poly)
    print_options = pygame_util.DrawOptions(screen)

    #floor segment (static)
    floor = pymunk.Segment(space.static_body, (0, 500), (800, 500), 10)
    floor.elasticity = 0.8
    floor.friction = 0.9
    space.add(floor)

    terrain_surface = pygame.Surface((600,600))
    terrain_surface.fill((pygame.color.THECOLORS["blue"]))
    #shape = pygame.draw.circle(terrain_surface, (255,255,255), (450,120), 100)


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
