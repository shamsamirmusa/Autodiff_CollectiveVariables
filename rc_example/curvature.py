import jax.numpy as jnp
from jax import grad, jit
import plumedCommunications as PLMD

plumedInit={"Value":PLMD.defaults.COMPONENT}
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

def r_g(x):
    grad_rc = grad(r_f)
    return grad_rc(x)


 

def cv(action:PLMD.PythonCVInterface):
    """
    This script defines a collective variable using the PYCVINTERFACE plugin.
    The CV represents the radius ofcurvature and it's derivatives wrt atomic positions.
    """

    zero_box_derivative = jnp.zeros((3, 3)) # place holder
    return r_f(action.getPositions()), r_g(action.getPositions()), zero_box_derivative




