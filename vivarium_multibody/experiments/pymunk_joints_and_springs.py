#Create a toy model using the pymunk pivot joint and damped spring
from typing import Optional, Tuple

import pymunk

from networkx.drawing import circular_layout
from pymunk import Space, pygame_util, Body, Circle, Vec2d, Segment
import sys
import pygame
import pymunk.pygame_util
import pymunk.constraints as constraints

def main(surface_anchor_world=None):


    #Initialize the Pygame Module and screen (where we draw the simulation result)
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    print_options = pygame_util.DrawOptions(screen)
    print_options.flags |= pymunk.SpaceDebugDrawOptions.DRAW_CONSTRAINTS

    #Space Object - global variable and assign it a gravity vector
    space = Space()
    space.gravity = 0, 981 #confirm units

    #Segment - Fixed ground for our object
    surface = pymunk.Segment(space.static_body, (0, 500), (600, 500), 100)
    surface.elasticity = 0.0
    surface.friction = 1.0
    top_y = surface.a.y - surface.radius


    #Dynamic body with mass, moment and position
    mass = 100
    a_local = (100, 0) #200 unit rod
    b_local = (-100, 0)
    radius = 20
    moment = pymunk.moment_for_segment(mass, a_local, b_local, radius)
    cell = Body(mass, moment)
    #Place rod so it just touches the surface
    cell.position = (200, 100-radius)

    #Create Circle shape and attach it to the body
    cell_shape = Segment(cell, a_local, b_local, radius)
    #cell_shape.sensor = True
    cell_shape.elasticity = 0.0
    cell_shape.friction = 0.8

    #Create tiny circle, pilus tip shape
    tip_mass = 5
    tip_radius = 10.0
    tip_moment = pymunk.moment_for_circle(tip_mass, 0, tip_radius)
    tip = pymunk.Body(tip_mass, tip_moment)
    attach_x = -80
    attach_y = cell_shape.radius
    rest_len = 5

    tip.position = cell.local_to_world((attach_x, attach_y + rest_len))
    tip_shape = Circle(tip, radius=tip_radius)
    tip_shape.elasticity = 0.0
    tip_shape.friction = 1.0

    #Connect pilus tip to the cell (PinJoint which is cell <--> pilus tip)
    #Internal tether that makes the pilus tip stay attached to the bottom of the cell
    #Fixes the distance between cell and tip, tip moves with the cell but can rotate or swing a little
    #OFF CENTER PinJoint allowed torque, so the rod rotated on impact
        #PinJoint only fixes the distance between the anchors but not their angle so the cell collapses because
        # the anchor was not at the center of mass (uncomment this to see the effects of PinJoint)
    cell_to_tip = pymunk.constraints.PinJoint(cell, tip, (attach_x, attach_y),(0,0))
    cell_to_tip.collide_bodies = False
    space.add(cell_to_tip)
    attached = False


    #Create a body for curli attachment - not added yet --> don't really need this
    curli_mass = 5
    curli_radius = 10.0
    curli_moment = pymunk.moment_for_circle(curli_mass, 0, curli_radius)
    curli = pymunk.Body(curli_mass, curli_moment)
    curli.position = ((cell.position.x + 5.0), (cell.position.y + 5.0))
    curli_shape = Circle(curli, radius=curli_radius)
    curli_shape.elasticity = 0.5
    curli_shape.friction = 0.5
    #space.add(curli, curli_shape)

    #want to make the pivot but keep it off until its on contact with surface
    anchor = (tip.position.x, top_y)
    tip_surface = pymunk.constraints.PivotJoint(tip, space.static_body, (0,0),anchor)
    tip_surface.collide_bodies = False
    tip_surface.max_force = 0
    tip_surface.max_torque = 0
    tip_surface.max_bias = 0
    space.add(tip_surface)

    #Create a Damped Spring (Curli)
    rod_anchor_local = (100, radius)
    rod_end_surface = cell.local_to_world(rod_anchor_local) #accounts for the rods position + location not just coordinates
    surface_anchor_world = (rod_end_surface.x, top_y)
    #need to add/figure out rest and rest length

    spring = pymunk.DampedSpring(cell, space.static_body, rod_anchor_local, surface_anchor_world, rest_length=rest_len,
                                 stiffness=0.1, damping=0.1)
    space.add(spring)


    attached = False

    #Add body, circle, and surface to the space
    space.add(cell, cell_shape, surface, tip, tip_shape)

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()

        # keep the static anchor until attachment is triggered
        if not attached:
            tip_surface.anchor_b = (tip.position.x, tip.position.y)

        # when the tip approaches the surface, bind it there
        if (not attached) and (tip.position.y >= top_y - tip_radius):
            tip_surface.anchor_b = (tip.position.x, top_y)  # snap static anchor to surface line
            tip_surface.max_force = float('inf')  #rigid weld at the pivot - no limit essentially
            tip_surface.max_bias = float('inf')
            attached = True

        screen.fill((255, 255, 255))
        space.step(0.001)
        clock.tick(60)
        space.debug_draw(print_options)
        pygame.display.flip()


if __name__ == '__main__':
        sys.exit(main())