import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from dolfinx import mesh, fem, io
import ufl
# from dolfinx.fem import petsc

from fenitop.topopt import topopt
from fenitop.utility import load_field_from_h5

domain = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [50, 50]],
                        [50, 50], CellType.quadrilateral)
u_opt, _ = load_field_from_h5(domain, "./data_optimize/u_field.h5")
u_ref, _ = load_field_from_h5(domain, "./data/u_reference.h5")

V0 = fem.FunctionSpace(domain, ("Lagrange", 1))
ux_opt = fem.Function(V0, name="ux"); ux_opt = u_opt.sub(0).collapse()
uy_opt = fem.Function(V0, name="uy"); uy_opt = u_opt.sub(1).collapse()

V0 = fem.FunctionSpace(domain, ("Lagrange", 1))
ux_ref = fem.Function(V0, name="ux"); ux_ref = u_ref.sub(0).collapse()
uy_ref = fem.Function(V0, name="uy"); uy_ref = u_ref.sub(1).collapse()


import math
import math
dx = ufl.dx(domain=domain)          # 仍用 reference-mesh 的测度

# --- 4.1 生成 DG0 指示函数 mask(x) --------------------------
V0 = fem.FunctionSpace(domain, ("DG", 0))
mask = fem.Function(V0, name="mask")
mask.x.array[:] = 1.0                 # 先全部置 1（参与积分）

# 找到“正方孔”内部的单元，把它们的 mask 设为 0
cell_mids = mesh.compute_midpoints(
    domain, domain.topology.dim,
    np.arange(domain.topology.index_map(domain.topology.dim).size_local,
              dtype=np.int32))

# 方孔中心 (25,25)，半边长 L/8 = 6.25
inside = np.where(
    (np.abs(cell_mids[:, 0] - 25.0) < 6.25) &
    (np.abs(cell_mids[:, 1] - 25.0) < 6.25)
)[0]
mask.x.array[inside] = 0.0            # 空洞区域不参与积分

# --- 4.2 带 mask 的 L²-平方积 -------------------------------
def l2_sq_masked(f):
    local = fem.assemble_scalar(fem.form(f * f * mask * dx))
    return domain.comm.allreduce(local, op=MPI.SUM)

# 误差分子 / 分母
num_x = l2_sq_masked(ux_opt - ux_ref)
den_x = l2_sq_masked(ux_ref)

num_y = l2_sq_masked(uy_opt - uy_ref)
den_y = l2_sq_masked(uy_ref)

ex = math.sqrt(num_x) / math.sqrt(den_x) if den_x > 0 else float("nan")
ey = math.sqrt(num_y) / math.sqrt(den_y) if den_y > 0 else float("nan")

if domain.comm.rank == 0:
    print(f"🔎  L2-relative errors (void excluded):  "
          f"ex = {ex*100:.2f} %,   ey = {ey*100:.2f} %")