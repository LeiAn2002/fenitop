"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType, compute_midpoints
from dolfinx import io
from dolfinx.fem import FunctionSpace, Function

from fenitop.topopt import topopt
from fenitop.utility import load_field_from_h5

L = 20.0
nel = 100

mesh_ref = create_rectangle(MPI.COMM_WORLD, [[-L / 2, -L / 2], [L / 2, L / 2]],
                            [nel, nel], CellType.quadrilateral)
if MPI.COMM_WORLD.rank == 0:
    mesh_ref = create_rectangle(MPI.COMM_WORLD, [[-L / 2, -L / 2], [L / 2, L / 2]],
                                [nel, nel], CellType.quadrilateral)
else:
    mesh_serial = None

msh_file = "./meshes/eye.msh"
u_ref, _ = load_field_from_h5(mesh_ref, "./data/u_reference.h5")

mesh, _, _ = io.gmshio.read_from_msh(
            msh_file, comm=MPI.COMM_WORLD, rank=0, gdim=2
        )

if MPI.COMM_WORLD.rank == 0:
    mesh_serial, _, _ = io.gmshio.read_from_msh(
            msh_file, comm=MPI.COMM_WORLD, rank=0, gdim=2
        )
else:
    mesh_serial = None

cell_dim = mesh_ref.topology.dim
cells_ref = np.arange(mesh_ref.topology.index_map(cell_dim).size_local, dtype=np.int32)

# mid‑points of reference and perforated cells (local arrays)
mids_ref = compute_midpoints(mesh_ref,  cell_dim, cells_ref)
mid_set_hole = {
    tuple(np.round(pt, 8))
    for pt in compute_midpoints(
        mesh,
        cell_dim,
        np.arange(mesh.topology.index_map(cell_dim).size_local, dtype=np.int32),
    )
}

V_mask = FunctionSpace(mesh_ref, ("DG", 0))
mask = Function(V_mask, name="mask")
mask_array = np.ones_like(mask.x.array, dtype=mask.x.array.dtype)

for local_id, pt in enumerate(mids_ref):
    if tuple(np.round(pt, 8)) not in mid_set_hole:
        mask_array[local_id] = 0.0  # this cell belongs to the hole
mask.x.array[:] = mask_array


def in_leaf_solid_zone(x: np.ndarray, *, L: float, enlarge: float = 1.0,
                       r0_factor: float = 0.35, k: float = 0.4) -> np.ndarray:

    # polar coordinates centred at (0, 0)
    r     = np.sqrt(x[0] ** 2 + x[1] ** 2)
    theta = np.arctan2(x[1], x[0])

    # teardrop in polar form  r_leaf(θ) = R₀·(1 − k·sinθ)
    R0     = r0_factor * L * enlarge
    r_leaf = R0 * (1.0 - k * np.sin(theta))

    return r <= r_leaf


fem = {  # FEA parameters
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 2000,
    "poisson's ratio": 0.34,
    # "disp_bc": lambda x: np.isclose(x[1], 0.0) | (np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)),
    # "traction_bcs": [[(0, 0.2),
    #                   lambda x: (np.isclose(x[1], 50))]],
    "traction_bcs": [],
    "body_force": (0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}

opt = {  # Topology optimization parameters
    "max_iter": 30,
    "opt_tol": 1e-2,
    # "vol_frac": 0.9,
    # "solid_zone": lambda x: (
    #     (np.less(x[0], -20.0) | np.greater(x[0], 20.0) |
    #      np.less(x[1], -20.0) | np.greater(x[1], 20.0))
    # ),

    # "void_zone": lambda x: (
    #     np.logical_and(
    #         np.abs(x[0]) < 6.25,
    #         np.abs(x[1]) < 6.25
    #     )
    # ),

    # "solid_zone_rho": lambda x: np.logical_not(
    #     np.logical_and(np.abs(x[0]) < 6.25,
    #                    np.abs(x[1]) < 6.25)
    # ),

    "void_zone": lambda x: np.full(x.shape[1], False),
    "solid_zone_rho": lambda x: np.full(x.shape[1], True),

    "solid_zone": lambda x: np.logical_not(
        in_leaf_solid_zone(x, L=L, enlarge=1.1)
    ),
    # "void_zone": lambda x: np.full(x.shape[1], False),
    # "solid_zone_rho": lambda x: np.full(x.shape[1], False),
    "theta_vec": mask,
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": 0.5,
    "beta_interval": 50,
    "beta_max": 128,
    "use_oc": False,
    "move": 0.01,
    "block_types": 5,
    "max_vf": 0.7,
    "u_ref_field": u_ref,
}

if __name__ == "__main__":
    topopt(fem, opt)

# Execute the code in parallel:
# mpirun -n 8 python3 scripts/mechanism_2d.py