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
from dolfinx.mesh import create_rectangle, CellType

from fenitop.topopt import topopt
from fenitop.utility import load_field_from_h5

mesh = create_rectangle(MPI.COMM_WORLD, [[-25.0, -25.0], [25, 25]],
                        [50, 50], CellType.quadrilateral)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(MPI.COMM_WORLD, [[-25.0, -25.0], [25, 25]],
                                   [50, 50], CellType.quadrilateral)
else:
    mesh_serial = None

u_ref, _ = load_field_from_h5(mesh, "./data/u_reference.h5")

fem = {  # FEA parameters
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 2.41,
    "poisson's ratio": 0.35,
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
    "max_iter": 600,
    "opt_tol": 1e-2,
    # "vol_frac": 0.9,
    # "solid_zone": lambda x: (
    #     (np.less(x[0], -20.0) | np.greater(x[0], 20.0) |
    #      np.less(x[1], -20.0) | np.greater(x[1], 20.0))
    # ),

    "void_zone": lambda x: (
        np.logical_and(
            np.abs(x[0]) < 6.25,
            np.abs(x[1]) < 6.25
        )
    ),

    "solid_zone_rho": lambda x: np.logical_not(
        np.logical_and(np.abs(x[0]) < 6.25,
                       np.abs(x[1]) < 6.25)
    ),
    "solid_zone": lambda x: np.full(x.shape[1], False),
    # "void_zone": lambda x: np.full(x.shape[1], False),
    # "solid_zone_rho": lambda x: np.full(x.shape[1], False),

    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": 0.5,
    "beta_interval": 40,
    "beta_max": 128,
    "use_oc": False,
    "move": 0.01,
    "block_types": 3,
    "max_vf": 0.7,
    "u_ref_field": u_ref,
}

if __name__ == "__main__":
    topopt(fem, opt)

# Execute the code in parallel:
# mpirun -n 8 python3 scripts/mechanism_2d.py