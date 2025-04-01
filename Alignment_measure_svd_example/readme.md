### A Measure of Straightness using SVD. 

This code calculates the "straightness" of a set of points in 3D space, using Principal Component Analysis (PCA) via Singular Value Decomposition (SVD). The function circumcenter_matrix computes the Bary circumcenters [1] for each triplet of points, and svd_fit performs SVD to extract singular values, and then calculates the ratio of the largest singular value to the sum of all singular values, representing the varience.  
This ratio serves as a measure of how linearly aligned the points are along a single direction.

## Requirements

- Jax numpy
- JAX jit
- PLUMED 2 with PythonCVInterface plugin

## Python Script (helix_length_cv.py)

- B_circumcenter(A, B, C): Calculates the circumcenter of a triangle defined by points A, B, and C.
- circumcenter_matrix(positions): Processes an array of positions to calculate the circumcenters of all consecutive triplets of atoms.
- distance(action): Interface function for PLUMED that retrieves positions via action.getPositions(), computes the circumcenters, and calculates the distance between the first and last circumcenter.

## PLUMED Input File (helix_length.dat)

Configures the PYCVINTERFACE to import the Python script and calculate the CV

# Running the Script

Ensure PLMED is compiled with support for PythonCVInterface.

# Plumed Driver command

plumed driver --plumed alignment_svd.dat --pdb 1kdx.pdb --mf_dcd 1kdx.dcd

# Output

- `cv.out` Containing the cv values
- `cvb1_der.out` Containing the autodiff and the Numerical differentiation.
- `cvb2_der.out` Containing the autodiff and the Numerical differentiation.  

----------------------------------------------------
[1] https://math.stackexchange.com/questions/1304194/center-of-circle-when-three-points-in-3-space-are-given/1304202#1304202