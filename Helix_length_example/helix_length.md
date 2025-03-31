### Helix Length Calculation
The project integrates jax with PLUMED through PythonCVInterface to calculate the Barycentric coordinates from which the distance between the first and last center is computed giving the length of a helix central axis. 
#### Requirements
- Jax numpy
- JAX jit
- PLUMED 2 with PythonCVInterface plugin
#### Python Script (helix_length_cv.py)
- B_circumcenter(A, B, C): Calculates the circumcenter of a triangle defined by points A, B, and C.  
- circumcenter_matrix(positions): Processes an array of positions to calculate the circumcenters of all consecutive triplets of atoms.
- distance(action): Interface function for PLUMED that retrieves positions via action.getPositions(), computes the circumcenters, and calculates the distance between the first and last circumcenter.

#### PLUMED Input File (helix_length.dat)

Configures the PYCVINTERFACE to import the Python script and calculate the CV  

#### Running the Script
Ensure PLMED is compiled with support for PythonCVInterface.  

##### Plumed Input File  
LOAD FILE=/Users/sss/Documents/EnergyGap_project/plumed2/plugins/pycv/PythonCVInterface.dylib  

hb1: GROUP ATOMS=1339,1353,1365,1376,1393,1415,1439,1463,1478,1497,1516,1527,1551   
hb2: GROUP ATOMS=1575,1589,1603,1624,1648,1670,1689,1708,1722,1734,1753,1764,1775,1787,1797  

len_b1: PYCVINTERFACE ATOMS=hb1 IMPORT=helix_length_cv CALCULATE=distance  
len_b2: PYCVINTERFACE ATOMS=hb2 IMPORT=helix_length_cv CALCULATE=distance  

PRINT FILE=helix_length.out ARG=* STRIDE=1  

##### Plumed Driver command
plumed driver --plumed helix_length.dat --pdb 1kdx.pdb --mf_dcd 1kdx.dcd  

#### Output
helix_length.out.reference containing the calculated distances for each frame.

Note! DISTANCE action requires ATOMS to contain only two atoms. make sure they are the first two indices specified in your GROUP ATOMS
