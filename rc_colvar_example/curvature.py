import jax.numpy as jnp
from jax import grad, jit
import plumedCommunications as PLMD

# Define the value and gradient functions
@jit
def r_f(x):
    r21 = x[0, :] - x[1, :]
    r23 = x[2, :] - x[1, :]
    r31 = x[2, :] - x[0, :]

    cos2theta = jnp.dot(r21, r23)**2 / (jnp.dot(r21, r21) * jnp.dot(r23, r23))
    sin2theta = 1 - cos2theta

    R2 = jnp.dot(r31, r31) / sin2theta / 4
    return jnp.sqrt(R2)

r_g = grad(r_f)

# Define the function for PLUMED
def r(x):
    return r_f(x), r_g(x)

# Function required by PYCV
def plumedCalculate(action: PLMD.PythonCVInterface):
    # Retrieve atomic positions passed by PLUMED
    positions = action.getPositions()
    
    value = r_f(positions)  # Value of the collective variable
    gradient = r_g(positions)  # Gradient of the collective variable

    # Pass results back to PLUMED
    action.setValue(value)  
    action.setDerivative(gradient)  











