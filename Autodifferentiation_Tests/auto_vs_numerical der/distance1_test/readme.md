## Distance Test

This is a test test for the Interface functionality. Here we compare a python function that computes the distance between two positions to that with plumed DISTANCE action. We also compare the values of jax autodiff with plumed NUMERIC_DERIVATIVES.

#### PLUMED DRIVER COMMAND
plumed driver --plumed distance1.dat --pdb 4dvd_plumed.pdb --mf_dcd 4dvd.dcd
