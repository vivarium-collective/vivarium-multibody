# Pymunk Biofilm: branch of viviarum-multibody repository 

This repository was branched to explore using multibody physics and pygame to model the steps of biofilm formation

 1. Added [pymunk_biofilm](https://github.com/vivarium-collective/vivarium-multibody/blob/biofilm-demo/vivarium_multibody/experiments/pymunk_biofilm.py)
    
* Experiment using the pymunk engine and architecture of the **mother machine**. In this script I try to simulate biofilm growth on a surface using the 2D physics of pymunk and multibody physics.

* Trying to simulate 4 cells growing on a surface and move according to multibody physics

2. Added [pymunk_exploration](https://github.com/vivarium-collective/vivarium-multibody/blob/biofilm-demo/vivarium_multibody/experiments/pymunk_exploration.py)

* Demo exploring pygame and pymunk from pygame tutorials. Initial exploration of pygame (without multibody_physics or mother machine structure).

3. Added [pymunk_joints_and_springs](https://github.com/vivarium-collective/vivarium-multibody/blob/biofilm-demo/vivarium_multibody/experiments/pymunk_joints_and_springs.py)

* Explored and added pygame pinjoint, pivotjoint, and damped spring to a single cell moving through space. The pinjoint was utilized to represent the fimbraie extension of the cell. The pivotjoint was used to represent the "bond" (weak) between the cell and the surface when the cell approaches the surface. The damped spring represents the attachment of curli to the surface.
  
* As a note, this was an exploration to see what joints and constraints were available to simulate a cell binding to the cell surface (reversible attachment of biofilm formation). Therefore, the friction, forces, and physics are not completely accurate and should be updated in the future if we want to use this. 


# Vivarium Project

Visit [the Vivarium Core
documentation](https://vivarium-core.readthedocs.io/) to learn how to
use the core Vivarium engine to create computational biology models.
Check out the
[getting started](https://vivarium-core.readthedocs.io/en/latest/getting_started.html)
guide of the documentation. 
