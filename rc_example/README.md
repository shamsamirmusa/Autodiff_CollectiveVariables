# Implementing the Radius of Curvature as a Collective Variable using PYCV

## **Introduction**
This repository builds upon the work described in the paper by Toni Giorgino [1],. This work integrates Python with PLUMED using the **PYCVINTERFACE** plugin, enabling enhanced sampling methods in molecular dynamics simulations.

The code provided implements a custom function (`r_f`) that calculates the **radius of curvature** of a circle passing through three atoms. 
Additionally, its gradient is computed using the JAX library. This CV interacts with PLUMED via the PyCvInterface directive. 

The Python module `plumedCommunications` includes the PythonCVInterface class, which provides access to simulation data such as:
- **Atomic positions**
- **Simulation step**
- **Returning CV values and derivatives back to PLUMED**

---

## **Implementation Components**
### **Python Script (`curvature.py`)**
- Loads the PythonCVInterface.
- Defines the radius of curvature (`r_f`) and its derivatives (`r_g`).
- Exports the functions to the PYCVINTERFACE plugin for use in a PLUMED workflow.

### **PLUMED Input File (`rc_input.dat`)**
- Configures the PYCVINTERFACE to import the Python script and calculate the CV.

---

## **Running the Simulation**
To run the simulation successfully:
1. Ensure `curvature.py` is in the same directory as the PLUMED input file (`rc_input.dat`) and your **trajectory file**.
2. Confirm that the shared library **PythonCVInterface.dylib** is accessible through your **PATH**.
3. Run the **PLUMED driver command**, for example below:

```bash
plumed driver --plumed plumed_input.dat --mf_xtc trajectory.xtc --timestep 0.002
```
## **Citation** 

- [1] Giorgino, (2019). PYCV: a PLUMED 2 Module Enabling the Rapid Prototyping of Collective Variables in Python. Journal of Open Source Software, 4(42), 1773, https://doi.org/10.21105/joss.01773
- [2] https://www.plumed-nest.org/consortium.html
