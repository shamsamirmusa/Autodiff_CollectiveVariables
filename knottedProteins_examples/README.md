# Writhe Function  

The writhe function is a mathematical measure of the self-entanglement of a curve in 3D space. It originates from the Gauss linking integral, which computes the topological linking between two curves.  
In the case of a single curve, the writhe describes how much the curve twists around itself.


$$
Wr(C) = \frac{1}{4\pi} \int_C \int_C \frac{(\dot{\gamma}(s) \times \dot{\gamma}(t)) \cdot (\gamma(s) - \gamma(t))}{|\gamma(s) - \gamma(t)|^3} \, ds \, dt
$$

where:  
$\gamma(ti)$ is the parameterized curve.  
$ \dot{\gamma}(ti) $ is the tangent vector.  
$\ (\dot{\gamma}(ti) \times \dot{\gamma}(tj)) $ is the **cross product** of tangents.  
$\ (\gamma(ti) - \gamma(tj)|^3 )$ normalizes the integral.



## **Kernel Function: Scalar Triple Product**
The scalar triple product is the core mathematical operation, which encodes the geometric twisting between two segments of the curve:

$$
K(i, j) = \frac{t_i \cdot (t_j \times (r_i - r_j))}{|r_i - r_j|^3}
$$

where:  

$t_i$ and $t_j$ are tangent vectors.  
$r_i$, $r_j$ are points on the curve. 





## **2-Simplex Restriction**
Instead of computing over all possible point pairs, the 2-simplex restriction was nused to consider segments rather than individual points.

The indices are restricted to ensure $i > j$, preventing redundant calculations.


Mathematically, this is done by setting a triangular matrix:

$$
i, j = \text{triuindices}(N - 1, k=1)
$$


## **Final Computation of the Writhe**
The final **writhe sum** is computed as:

$$
Wr = \frac{1}{2\pi} \sum_{i > j} \frac{t_i \cdot (t_j \times (r_i - r_j))}{|r_i - r_j|^3}
$$

where the **sum runs over all distinct segment pairs**.

## **Citation**

- Røgen, P., & Fain, B. (2003). Automatic classification of protein structure by using Gauss integrals. Proceedings of the National Academy of Sciences, 100(1), 119-124.
