# Overall Project Goal

The project aims to study how properties of an underlying PDE and its discretization influence the structure of the resulting linear system and, in turn, which preconditioner is most effective for solving that system iteratively. We will compare PDE-level and discretization-level features—such as equation type, coefficient, mesh size, time step, and stencil, with resulting matrix-level features such as sparsity, symmetry, conditioning, and spectral behavior. The main goal is to determine which (inexpensive) features best predict a near-optimal solver–preconditioner configuration, and whether incorporating upstream PDE information provides useful predictive power beyond matrix properties alone. A longer-term goal is to identify relationships that generalize across problem sizes, parameter regimes, and possibly discretization methods.

# Concrete Goals for Final Submission:

As a preliminary exploration, we will varying the variables discussed above for the Heat Equation, Advection-diffusion-reaction Equation, and Burgers Equation. As a stretch goal, we will utilize AI agents to extend our framework to broader families of PDEs.

## Preconditioners
- SPAI
- ILU
- Jacobi
- Block Jacobi
- Truncated-Neumann Jacobi
- Multigrid preconditioners

## PDEs

### Heat Equation

- Stencils: 5-point, 9-point cross, 9-point box; vary width, time, 
- RHS (f): TBD
- Anisotropy parameter


### Advection–diffusion–reaction
- Details TBD

### Burgers Equation
- Details TBD
