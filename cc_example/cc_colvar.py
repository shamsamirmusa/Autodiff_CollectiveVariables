
import jax.numpy as jnp
from jax import grad, jit
import plumedCommunications as PLMD

plumedInit = {"Value": PLMD.defaults.COMPONENT} # ask toreturn a vector

@jit
def center_of_curvature(x):
    p1, p2, p3 = x[0, :], x[1, :], x[2, :]

    # Midpoints of segments
    mid12 = (p1 + p2) / 2
    mid23 = (p2 + p3) / 2

    # Perpendicular bisectors
    v12 = p2 - p1
    v23 = p3 - p2
    perp12 = jnp.cross(v12, jnp.cross(v12, v23))
    perp23 = jnp.cross(v23, jnp.cross(v23, v12))

    # Normalize to get the unit vector, the dirction is what we need here
    perp12 = perp12 / jnp.linalg.norm(perp12)
    perp23 = perp23 / jnp.linalg.norm(perp23)

    # Formulate the linear system
    A = jnp.vstack([perp12, perp23])
    b = jnp.array([jnp.dot(mid12, perp12), jnp.dot(mid23, perp23)])

    # Solve for center
    center = jnp.linalg.lstsq(A, b, rcond=None)[0]
    return center

# Separate functions for each component
def center_x(action: PLMD.PythonCVInterface):
    positions = action.getPositions()
    zero_box_derivative = jnp.zeros((3, 3))  # Placeholder  
    x_value = center_of_curvature(positions)[0]
    grad_x = grad(lambda x: center_of_curvature(x)[0])(positions)
    return x_value, grad_x, zero_box_derivative

def center_y(action: PLMD.PythonCVInterface):
    positions = action.getPositions()
    zero_box_derivative = jnp.zeros((3, 3))  # Placeholder
    y_value = center_of_curvature(positions)[1]
    grad_y = grad(lambda x: center_of_curvature(x)[1])(positions)
    return y_value, grad_y, zero_box_derivative

def center_z(action: PLMD.PythonCVInterface):
    positions = action.getPositions()
    zero_box_derivative = jnp.zeros((3, 3))  # Placeholder
    z_value = center_of_curvature(positions)[2]
    grad_z = grad(lambda x: center_of_curvature(x)[2])(positions)
    return z_value, grad_z, zero_box_derivative
