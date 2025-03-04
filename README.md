# **Collective Variables in Molecular Dynamics**


## **This Repository Contains**
- **Differentiated Collective Variable Examples**: Implementations of CVs with automatic differentiation into PLUMED using PYMC [1].
- **JAX Auto-Diff Tests**: Edge-case testing on operations such as `max`, `min`, and `sort` to ensure correct differentiation behavior.
  

## **Overview**
Collective variables (CVs) are structural parameters that can be computed throughout a molecular dynamics (MD) simulation. They serve as mathematical functions that map high-dimensional atomic coordinates into a lower-dimensional representation, capturing key aspects of the system's state or progress along a reaction pathway.

### **Why Create and Use Collective Variables?**
- **To Simplify Complex Dynamics**: Instead of analyzing all atomic coordinates, CVs focus on essential features that describe the system's behavior.
- **To Quantify Structural Changes**: They provide insight into meaningful transformations, such as conformational shifts.

### **CVs in Enhanced Sampling & PLUMED Integration**
In MD simulations, the system evolves according to variations in the energy landscape, which is determined by the force field. During a simulation, inorder to fastforward to the crucial moments, these simulations are biased using CVs. The variation of the CV within the new landscape must be accounted for inorder to  calculate the forces accurately.

In this work, JAX, known  for its efficient, vectorized, and parallel computation of CV gradients across all frames, is used. 
The CVs are implemented within PLUMED [2] enhanced sampling methods via PYMC, a Python-based interface that enables rapid prototyping of new CVs for trajectory analysis or biasing functions.

## **Citation**  
- [1] Giorgino, (2019). PYCV: a PLUMED 2 Module Enabling the Rapid Prototyping of Collective Variables in Python. Journal of Open Source Software, 4(42), 1773, https://doi.org/10.21105/joss.01773
- [2] Tribello, G. A., Bonomi, M., Branduardi, D., Camilloni, C., & Bussi, G. (2014). PLUMED 2: New feathers for an old bird. Computer physics communications, 185(2), 604-613. 
