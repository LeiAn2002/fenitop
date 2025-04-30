# -*- coding: utf-8 -*-
"""
Mechanical-displacement cloak – pre-processing (dolfinx-0.7.3)
生成四个 XDMF 文件：
  ux_reference / uy_reference / ux_hole / uy_hole
可以直接被 ParaView 打开。
"""

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem import petsc

# ----------------------- 0. 参数 -----------------------
L, nel, order = 50.0, 50, 1           # 外形尺寸 / 网格
hole_len = L / 4
E, nu = 2.41, 0.35                    # 材料参数
void_fact = 1e-8                      # 空洞等效极软因子


def solve_case(with_hole: bool):
    # ---------- 1. 网格 ----------
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, [[0.0, 0.0], [L, L]],
        [nel, nel], cell_type=mesh.CellType.quadrilateral)

    # ---------- 2. 向量位移空间 ----------
    V_vec = fem.FunctionSpace(
        domain,
        ufl.VectorElement("Lagrange", domain.ufl_cell(), order, dim=2))

    u  = ufl.TrialFunction(V_vec)
    v  = ufl.TestFunction(V_vec)
    uh = fem.Function(V_vec, name="u")      # 待求解位移向量场

    # ---------- 3. 实/空洞指示函数 ----------
    chi = fem.Function(fem.FunctionSpace(domain, ("DG", 0)))
    chi.x.array[:] = 1.0
    if with_hole:
        mids = mesh.compute_midpoints(
            domain, domain.topology.dim,
            np.arange(domain.topology.index_map(domain.topology.dim).size_local,
                      dtype=np.int32))
        xmin, xmax = (L-hole_len)/2, (L+hole_len)/2
        mask = ((mids[:, 0] >= xmin) & (mids[:, 0] <= xmax) &
                (mids[:, 1] >= xmin) & (mids[:, 1] <= xmax))
        chi.x.array[np.where(mask)[0]] = void_fact

    mu    = fem.Function(chi.function_space)
    lam   = fem.Function(chi.function_space)
    mu.x.array[:]  = E/(2*(1+nu))           * chi.x.array
    lam.x.array[:] = E*nu/((1+nu)*(1-2*nu)) * chi.x.array

    def sigma(w):
        eps = ufl.sym(ufl.grad(w))
        return 2*mu*eps + lam*ufl.tr(eps)*ufl.Identity(2)

    a  = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    Lf = ufl.inner(fem.Constant(domain, PETSc.ScalarType((0.0, 0.0))), v) * ufl.dx

    # ---------- 4. 边界条件 ----------
    # uy = 0 底边
    f_bot = mesh.locate_entities_boundary(domain, 1,
        lambda x: np.isclose(x[1], 0.0))
    bc_bot = fem.dirichletbc(PETSc.ScalarType(0.0),
        fem.locate_dofs_topological(V_vec.sub(1), 1, f_bot), V_vec.sub(1))

    # ux = 0 左下角
    f_corner = mesh.locate_entities_boundary(domain, 0,
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    bc_corner = fem.dirichletbc(PETSc.ScalarType(0.0),
        fem.locate_dofs_topological(V_vec.sub(0), 0, f_corner), V_vec.sub(0))

    # uy = 2 mm 顶边
    f_top = mesh.locate_entities_boundary(domain, 1,
        lambda x: np.isclose(x[1], L))
    bc_top = fem.dirichletbc(
        fem.Constant(domain, PETSc.ScalarType((0.0, 2.0))),
        fem.locate_dofs_topological(V_vec, 1, f_top), V_vec)

    # ---------- 5. 求解 ----------
    problem = petsc.LinearProblem(
        a, Lf, bcs=[bc_bot, bc_corner, bc_top],
        u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem.solve()

    # ---------- 6. 拆分位移分量 ----------
    V_scal = fem.FunctionSpace(domain, ("Lagrange", order))  # 与向量空间共节点
    ux = fem.Function(V_scal, name="ux")
    uy = fem.Function(V_scal, name="uy")

    # 向量 DOF 排序：node0-ux, node0-uy, node1-ux, node1-uy, ...
    ux.x.array[:] = uh.x.array[0::2]
    uy.x.array[:] = uh.x.array[1::2]

    return domain, ux, uy


# ------------------- 生成两种结构 -------------------
mesh_ref,  ux_ref,  uy_ref  = solve_case(False)
mesh_hole, ux_hole, uy_hole = solve_case(True)

# ------------------- 写入 XDMF ---------------------
def write(path, m, f):
    with io.XDMFFile(m.comm, path, "w") as X:
        X.write_mesh(m)
        X.write_function(f)

write("./data/ux_reference.xdmf", mesh_ref,  ux_ref)
write("./data/uy_reference.xdmf", mesh_ref,  uy_ref)
write("./data/ux_hole.xdmf",     mesh_hole, ux_hole)
write("./data/uy_hole.xdmf",     mesh_hole, uy_hole)

if mesh_ref.comm.rank == 0:
    print("✅ displacement fields saved — open .xdmf in ParaView.")

# ------------------- 4. 计算 L2 误差 -------------------
import math
import math
dx = ufl.dx(domain=mesh_ref)          # 仍用 reference-mesh 的测度

# --- 4.1 生成 DG0 指示函数 mask(x) --------------------------
V0 = fem.FunctionSpace(mesh_ref, ("DG", 0))
mask = fem.Function(V0, name="mask")
mask.x.array[:] = 1.0                 # 先全部置 1（参与积分）

# 找到“正方孔”内部的单元，把它们的 mask 设为 0
cell_mids = mesh.compute_midpoints(
    mesh_ref, mesh_ref.topology.dim,
    np.arange(mesh_ref.topology.index_map(mesh_ref.topology.dim).size_local,
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
    return mesh_ref.comm.allreduce(local, op=MPI.SUM)

# 误差分子 / 分母
num_x = l2_sq_masked(ux_hole - ux_ref)
den_x = l2_sq_masked(ux_ref)

num_y = l2_sq_masked(uy_hole - uy_ref)
den_y = l2_sq_masked(uy_ref)

ex = math.sqrt(num_x) / math.sqrt(den_x) if den_x > 0 else float("nan")
ey = math.sqrt(num_y) / math.sqrt(den_y) if den_y > 0 else float("nan")

if mesh_ref.comm.rank == 0:
    print(f"🔎  L2-relative errors (void excluded):  "
          f"ex = {ex*100:.2f} %,   ey = {ey*100:.2f} %")