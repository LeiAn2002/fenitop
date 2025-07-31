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
from dolfinx.fem import FunctionSpace, Function, VectorFunctionSpace

from fenitop.topopt import topopt
from fenitop.utility import load_field_from_h5

L = 20.0
nel = 200

mesh_ref = create_rectangle(MPI.COMM_WORLD, [[-L / 2, -L / 2], [L / 2, L / 2]],
                            [nel, nel], CellType.quadrilateral)
if MPI.COMM_WORLD.rank == 0:
    mesh_ref_serial = create_rectangle(MPI.COMM_WORLD, [[-L / 2, -L / 2], [L / 2, L / 2]],
                                [nel, nel], CellType.quadrilateral)
else:
    mesh_ref_serial = None

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


def inside_leaf(x: np.ndarray,               # (..., 2) 数组，最后一维是 (x, y)
                W: float,                     # 叶子最大宽度（左右最外缘距离）
                L: float,                     # 叶子总长度（顶点到尾端距离）
                centre: tuple[float, float] | None = None,
                p: float = 2.0               # 侧边“圆滑度/尖锐度”控制参数
                ) -> np.ndarray:
    if x.shape[0] < 2:
        raise ValueError("预期 x 为 (3, N) 或至少 (2, N) 的坐标数组。")

    # ------------------------------------------------------------
    # 1. 平移使叶片中心位于 (0,0)
    # ------------------------------------------------------------
    if centre is None:
        cx = cy = 0.0
    else:
        cx, cy = centre

    xs = x[0] - cx           # shape: (N,)
    ys = x[1] - cy           # shape: (N,)

    # ------------------------------------------------------------
    # 2. 垂直方向范围检查
    # ------------------------------------------------------------
    inside_vertical = np.abs(ys) <= L / 2

    # ------------------------------------------------------------
    # 3. 计算当前 y 高度允许的半宽度
    # ------------------------------------------------------------
    y_norm = np.abs(ys) / (L / 2)             # 归一化到 [0,1]
    half_width = (W / 2) * (1.0 - y_norm**p)  # shape: (N,)
    half_width = np.clip(half_width, 0.0, None)

    # ------------------------------------------------------------
    # 4. 判断左右边界
    # ------------------------------------------------------------
    inside_horizontal = np.abs(xs) <= half_width

    # ------------------------------------------------------------
    # 5. 组合结果：同时满足垂直与水平条件
    # ------------------------------------------------------------
    return inside_vertical & inside_horizontal


fem = {  # FEA parameters
    "mesh": mesh,
    "mesh_ref": mesh_ref,
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
    "max_iter": 300,
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
    # "U_ref_vec": U_ref_vec,
    "void_zone": lambda x: np.full(x.shape[1], False),
    "solid_zone_rho": lambda x: np.full(x.shape[1], True),

    "solid_zone": lambda x: np.logical_not(
        inside_leaf(x, W=0.8*L, L=L, p=2.2)
    ),
    # "void_zone": lambda x: np.full(x.shape[1], False),
    # "solid_zone_rho": lambda x: np.full(x.shape[1], False),
    # "theta_vec": mask,
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