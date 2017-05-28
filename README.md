# CUDA-Lattice-Boltzmann-Hydrodynamic-Simulator
It is an illustration/education/experimental program for CUDA parallel computation on 3D Lattice Boltzmann hydrodynamic simulation.


## Lattice Boltzmann Method(LBM) - 3D hydrodynamic simulation
LBM is a lattice automata computatinoal fluid dynamics simulation algorithm for solving Navier-Stokes equation.  The equation can be used to describe aerodynamics or hydrodynamics systems.  It can also describe gas/liquid/surface interaction, as in the case of contact angle of a dropplet on a surface.

This LBM simulator is conditioned to simulate a liquid droplet moving on two parallel surfaces.  The advancing and receeding angle of a moving droplet is measured as a function of surface topology.  The core is general, which can be adapted to simulate any hydro/aero-dynamics.  Currently, the output is a 2D cross-sectional view of a droplet in ppm format.  
This program is for experimentation and demostration only, and would require further work to handle large 3D data properly.

The LBE model employed here is D4Q19 b=24.


## Requirment
Nvidia CUDA compatible video card
CUDA SDK 7.0+
CMake 2.8+

## Build Instruction
Typical cmake project.  You can build with Cmake-GUI.  One can choose their favorite compiler target.  It is tested in VS2012 & VS2015, but should work in any compiler and compatible platform.
To build with cmake command line




