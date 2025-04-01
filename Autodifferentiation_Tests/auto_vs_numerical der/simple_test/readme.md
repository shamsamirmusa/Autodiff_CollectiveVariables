# Custom Collective Variable Implementation

## Overview

This document examines 
1. The functionality of the Interface.
2. `INIT` keyword in a computational setup using PLUMED.  
3. Jax autodeff.  


This Python module takes the $x$, $y$, $z$ components of the first position and calculates $cv$ = $2x^{2} + 2y^{2} + 2z^{2}$.

### Expected Output  

The expected output by the function is confirmed by manual calculation, is as follows: 

| Quantity (nm)  | Value |
| -------------  | ------------- |
|   cv    | 43.21|
| dcv/dx  | -16.04 |
| dcv/dy  | 6.36 |
| dcv/dz  | 6.90 |

## Initialization

In Plumed, *plumedInit* dictionary defines the properties of the custom CV particularly the periodicity and whether their derivative should be computed. 
----------------------------------------------------------
#### Example use 1: 
##### No periodicity.

plumedInit = {
    "COMPONENTS": {
        # "Distance": {"period": None , "derivative": True},}}

By specifying period: None, PLUMED treats the collective variable as non-periodic, meaning the values are not wrapped or repeated within a defined interval (such as 0 to 360 degrees for angles, where sin(90°) = sin(450°)). Instead, it allows the CV values to increase or decrease freely without periodic constraints.   
And infact, when the above plumedInit is applied, the result matches the calculation by hand to few significant figures.  

| Quantity (nm)  | Value |
| -------------  | ------------- |
|   cv    | 43.213757|
| dcv/dx  | -16.0499992371|
| dcv/dy  | 6.3615999222 |
| dcv/dz  | 6.9019994736 |
-------------------------------------------------------
#### Example us 2: 
##### With Periodicity Consideration

If the periodicity of the function must be considered, it can be implemented as follows

`plumedInit = {`
    `"COMPONENTS": {`
        `# "Distance": {"period": ["0", 1.3] ,` `"derivative": True},}}`

Here the collective variable (CV) values are wrapped within the interval [0, 1.3].

| Quantity (nm)  | Value |
| -------------  | ------------- |
|   cv    | 0.313757|
| dcv/dx  | -16.0499992371|
| dcv/dy  | 6.3615999222 |
| dcv/dz  | 6.9019994736 |
  
-------------------------------------------------------------
#### Example use 3: 
##### Overriding Periodicity in PLUMED Input

Removing periodicity could also be done in the plumed input file but it seems to be overriden by what is set in the plumedInit object.  
For example, if we kept the plumedInit in the python module as follows  

`plumedInit = {`
    `"COMPONENTS": {`
        `"Distance": {"period": ["0", 1.3] , "derivative": True},}}  `

and *NOPBC* was included in th plumed input like this   

`cv: PYCVINTERFACE ATOMS=ca_atoms IMPORT=simple_test CALCULATE=scratch_fn NOPBC`

The cv output would still be restrained between 0 and 1.3

-------------------------------------------------
#### Example use 4:  
##### Comparing Jax Auto-Diff to PLUMED Numerical Derivatives

To inspect jax auto diff functionality we can compare it to plumed NUMERICAL_DERIVATIVES by creating a new label named cv_N for example.  

cv_N: PYCVINTERFACE ATOMS=ca_atoms IMPORT=simple_test CALCULATE=scratch_fn NUMERICAL_DERIVATIVES  
DUMPDERIVATIVES ARG=cv_N STRIDE=1 FILE=simple_NUM.out
 

| jax autodiff  |  Numerical derivative|
| -------------  | ------------- |
|  -16.0499992371  | -16.0499997139|
| 6.3615999222  | 6.3615999222|
| 6.9019994736  | 6.9019999504 |


- Plumed Driver command

The plumed Driver command is 
`plumed driver --plumed simple_test.dat --pdb 4dvd_plumed.pdb --mf_dcd 4dvd.dcd`  

-A Note on units  

Plumed does the calculation in A but prints them as nm. 
Try  
`plumed driver --plumed simple_test.dat --pdb 4dvd_plumed.pdb --mf_dcd 4dvd.dcd  --Length = A`
