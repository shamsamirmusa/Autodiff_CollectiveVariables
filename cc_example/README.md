# The Center of Curvature Collective Variable

## Introduction
This code defines a **collective variable** for finding the **center of curvature**.  
The **center of the circle** is the intersection of the **perpendicular bisectors** of any two segments formed by three points.  
Below is the **mathematical derivation**.

---

## **Theory**  

Given three points **$\mathbf{p}_1$**, **$\mathbf{p}_2$**, and **$\mathbf{p}_3$**,  
the **line segments** $\mathbf{v}$ can be constructed as:

$$
\mathbf{v}_{12} = \mathbf{p}_2 - \mathbf{p}_1
$$

$$
\mathbf{v}_{23} = \mathbf{p}_3 - \mathbf{p}_2
$$


To calculate the **perpendicular vectors** to these bisectors, we first define **$\mathbf{N}$**,  
a vector **perpendicular to the plane** formed by **$\mathbf{v}_{12}$** and **$\mathbf{v}_{23}$**:  


$$
\mathbf{N} = \mathbf{v}_{12} \times \mathbf{v}_{23}
$$

Now, the **perpendicular bisector vectors** **$\mathbf{v}_{12}^\perp$** and **$\mathbf{v}_{23}^\perp$** are calculated as:

$$
\mathbf{v}_{12}^\perp = \mathbf{v}_{12} \times \mathbf{N}
$$

$$
\mathbf{v}_{23}^\perp = \mathbf{v}_{23} \times \mathbf{N}
$$



The **normalized perpendicular vectors** are given by:

$$
\mathbf{v}_{12}^\perp = \frac{\mathbf{v}_{12}^\perp}{\|\mathbf{v}_{12}^\perp\|}
$$
$$
\mathbf{v}_{23}^\perp = \frac{\mathbf{v}_{23}^\perp}{\|\mathbf{v}_{23}^\perp\|}
$$

The **parametric equation** of the bisector to the line segments has the form:

$$
\mathbf{x} = \mathbf{m}_{12} + t_1 \mathbf{v}_{12}^\perp
$$

$$
\mathbf{x} = \mathbf{m}_{23} + t_2 \mathbf{v}_{23}^\perp
$$

where:
- **$ \mathbf{m}_{12} $** and **$ \mathbf{m}_{23} $** are the **midpoints** of the segments.
- **$ \mathbf{x} $** is the **center of curvature**.

The relationship between the perpendicular bisectors and the midpoints is given by:

$$
t_1 \mathbf{v}_{12}^\perp - t_2 \mathbf{v}_{23}^\perp = \mathbf{m}_{23} - \mathbf{m}_{12}
$$

This can be rewritten in **matrix form** as:

$$
\mathbf{A} \cdot \mathbf{t} = \mathbf{b}
$$

and solved using the **least square method**.

