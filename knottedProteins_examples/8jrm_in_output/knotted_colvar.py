import jax.numpy as jnp
from jax import grad, jit, vmap
import plumedCommunications as PLMD
import jax
jax.config.update("jax_enable_x64", True) 

plumedInit = {
"Value" : {"period":None, "derivative":True}

}

@jit
def compute_writhe(positions):
    N = positions.shape[0]
    tangents = jnp.diff(positions, axis=0)
    seg_len = jnp.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / seg_len

    def pairwise_writhe(i1, i2):
        ri = positions[i1]
        rj = positions[i2]
        ti = tangents[i1]
        tj = tangents[i2]
        diff = ri - rj
        cross_prod = jnp.cross(tj, diff)
        scalar_triple = jnp.dot(ti, cross_prod)
        dist_cubed = jnp.linalg.norm(diff) ** 3
        return (scalar_triple / dist_cubed) * seg_len[i1] * seg_len[i2]

    # Create indices for all i1 < i2 combinations
    i1_indices, i2_indices = jnp.triu_indices(N - 1, k=1)

    # Vectorize pairwise_writhe over index arrays
    vectorized_writhe = vmap(pairwise_writhe)      
    writhe_sum = jnp.sum(vectorized_writhe(i1_indices, i2_indices))

    writhe = (1.0 / (2.0 * jnp.pi)) * writhe_sum
    return writhe

writhe_g = jit(grad(compute_writhe)) 

zero_box_derivative = jnp.zeros((3, 3)) # place holder



def cv(action:PLMD.PythonCVInterface):
    """
    This script defines a collective variable using the PYCVINTERFACE plugin.
    The CV represents the space writhe of a curve.
    """
    x=action.getPositions()
    writhe_value = compute_writhe(x)
    writhe_grade = writhe_g(x)
    return writhe_value, writhe_grade, zero_box_derivative