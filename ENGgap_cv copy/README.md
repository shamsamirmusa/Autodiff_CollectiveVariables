# The Energy Gap Method CV


The following document includes two sections:  
- Theory behind the code.  
- How to use the code.

---

## The Theory behind the code

The dynamic existence of a protein can be tracked through snapshots, or "frames," captured at specific time steps. At each frame, residues are found in unique positions—with some cooperating towards protein folding, others resisting the fold and creating tension—while the role of some remains enigmatic.

This work operates under the premise that a protein's destiny is influenced by its constituent residues. By employing the energy gap method, the **ENGap Collective Variable** was created to be used as a biasing potential integrated into PLUMED.

The energy gap is defined as the difference between the least and second least eigenvalues, normalized by the average difference between consecutive eigenvalues. Each frame will have an Energy Gap value, **ENG(t)**, and its standard deviation, **SDENG(t)**. Frames with larger ENG(t) and smaller SDENG(t) may reflect a native protein state [Meli et al.](#meli2020).

### Energy Gap Definitions

**ENG(t)** is defined as:

$$
\text{ENG}(t) = \frac{\Delta \lambda_{1-2}(t)}{\langle \Delta \lambda(t) \rangle}
$$

and the spectral standard deviation is given by:

$$
\text{SDENG}(t) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\lambda_i - \bar{\lambda}\right)^2}
$$

#### ENGap Collective Variable Computation

The Collective Variable (CV) is computed as:

$$
\text{CV}(t) = \alpha(t) \cdot \text{ENG}(t) - \beta(t) \cdot \text{SDENG}(t)
$$

where **\(\alpha(t)\)** and **\(\beta(t)\)** are weights modulating **ENG(t)** and **SDENG(t)**, respectively. These weights are dynamically adjusted based on the computed ENG and SDENG values:

- \(\alpha = \alpha \times 1.1\) if \(\text{ENG}(t) > \text{threshold}_{\text{ENG}}\)
- \(\beta = \beta \times 0.9\) if \(\text{SDENG}(t) < \text{threshold}_{\text{SDENG}}\)

The threshold values are defined in [Meli et al] [1]

To integrate this CV into a biasing potential in PLUMED, it first needs to be differentiated with respect to the atomic positions \(\mathbf{r}\):

$$
\frac{\partial \text{CV}}{\partial \mathbf{r}} = \alpha(t) \cdot \frac{\partial \text{ENG}}{\partial \lambda_i} \cdot \frac{\partial \lambda_i}{\partial M} \cdot \frac{\partial M}{\partial D} \cdot \frac{\partial D}{\partial \mathbf{r}} - \beta(t) \cdot \frac{\partial \text{SDENG}}{\partial \lambda_i} \cdot \frac{\partial \lambda_i}{\partial M} \cdot \frac{\partial M}{\partial D} \cdot \frac{\partial D}{\partial \mathbf{r}}
$$

**Where:**

- **\(\alpha\):** Weight modulating the influence of ENG, indicating its sensitivity to eigenvalues.
- **\(\beta\):** Weight modulating the influence of SDENG, indicating its sensitivity to eigenvalues.
- **\(\lambda_i\):** The eigenvalue.
- **\(M\):** The energy matrix computed using Lennard-Jones and Coulombic interaction models.
- **\(D\):** The standard Euclidean distances matrix.

---

## How to use the code

This code is designed to analyze molecular dynamics simulation data by processing GROMACS-generated TPR and XTC files using the MDAnalysis library. The core functionality is based on extracting protein structure and atomic properties from these files and converting them into arrays suitable for efficient computation with JAX.

### Code's Input Files

To use the code, the following files are needed:

```plaintext
tpr_file = "/path/to/your/tpr/file"
xtc_file = "/path/to/your/xtc/file"
lj_contents = "/path/to/your/tpr_contents.txt"
```
## **Citation**
- Meli, M., Morra, G., & Colombo, G. (2020). Simple model of protein energetics to identify ab initio folding transitions from all-atom MD simulations of proteins. Journal of Chemical Theory and Computation, 16(9), 5960-5971.
