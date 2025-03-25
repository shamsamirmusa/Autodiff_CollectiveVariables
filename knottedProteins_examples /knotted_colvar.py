import jax.numpy as jnp
from jax import grad, jit
import plumedCommunications as PLMD

@jit
def compute_writhe_simplex(positions:float)->float:
    """
    Compute the writhe of a protein backbone using the 2-simplex formulation.

    Parameters:
        positions: jnp.array of shape (N, 3), where N is the number of residues.
        
    Returns:
        Writhe value (scalar)
    """
    N = positions.shape[0]

    # Compute tangent vectors (finite difference approximation)
    tangents = jnp.diff(positions, axis=0)
    tangents /= jnp.linalg.norm(tangents, axis=1, keepdims=True)  # Normalize

    # Generate all index pairs (i, j) with i > j (strictly upper triangular part)
    i_idx, j_idx = jnp.triu_indices(N - 1, k=1)  # k=1 ensures i > j

    # Extract relevant points and tangents
    r_i, r_j = positions[i_idx], positions[j_idx]
    t_i, t_j = tangents[i_idx], tangents[j_idx]

    # Compute pairwise displacement vectors
    diff = r_i - r_j  # Shape: (M, 3)

    # Compute cross product t_j x (r_i - r_j)
    cross_prod = jnp.cross(t_j, diff)  # Shape: (M, 3)

    # Compute scalar triple product: t_i ⋅ (t_j × (r_i - r_j))
    scalar_triple_product = jnp.einsum("ij,ij->i", t_i, cross_prod)
    # scalar_triple_product = jnp.dot(t_i, cross_prod)


    distance_cubed = jnp.linalg.norm(diff, axis=1) ** 3
    distance_cubed = jnp.where(distance_cubed == 0, jnp.inf, distance_cubed)  # Avoid division by zero

    # Compute writhe sum witht he restriction.
    writhe_sum = jnp.sum(scalar_triple_product / distance_cubed)

    # Finall, with the normalization factor
    writhe = (1 / (2 * jnp.pi)) * writhe_sum  # Factor adjusted for simplex
   

    return writhe


writhe_grade = jit(grad(compute_writhe_simplex)) 

zero_box_derivative = jnp.zeros((3, 3)) # place holder

plumedInit = {
"Value" : {"period":None, "derivative":True}

}

def cv(action:PLMD.PythonCVInterface):
    """
    This script defines a collective variable using the PYCVINTERFACE plugin.
    The CV represents the radius ofcurvature and it's derivatives wrt atomic positions.
    """
    x=action.getPositions()
    writhe_value = compute_writhe_simplex(x)
    writhe_grade = writhe_grade(x)
    return writhe_value, writhe_grade, zero_box_derivative