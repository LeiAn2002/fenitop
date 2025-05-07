# -*- coding: utf-8 -*-
"""
Pre-processing for mechanical cloak ― force-controlled version
Produces ux_/uy_ fields for solid plate and plate-with-hole.
"""

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem import petsc

# ---------- parameters ----------
L, nel, order = 50.0, 50, 1
hole_len = L / 4
E, nu = 2.41, 0.35
void_fact = 1e-8
ty = 0.2                         # uniform traction (N/mm) on top edge


def solve_case(with_hole: bool):
    # 1. mesh -------------------------------------------------------------
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, [[0, 0], [L, L]],
        [nel, nel], cell_type=mesh.CellType.quadrilateral)

    # 2. function spaces --------------------------------------------------
    V = fem.FunctionSpace(domain,
        ufl.VectorElement("Lagrange", domain.ufl_cell(), order, dim=2))
    u, v  = ufl.TrialFunction(V), ufl.TestFunction(V)
    uh    = fem.Function(V, name="u")

    # 3. material (same as before) ---------------------------------------
    chi = fem.Function(fem.FunctionSpace(domain, ("DG", 0)))
    chi.x.array[:] = 1.0
    if with_hole:
        mids = mesh.compute_midpoints(domain, domain.topology.dim,
            np.arange(domain.topology.index_map(domain.topology.dim).size_local,
                                                       dtype=np.int32))
        half = hole_len/2
        mask = (np.abs(mids[:,0]-L/2)<half) & (np.abs(mids[:,1]-L/2)<half)
        chi.x.array[np.where(mask)[0]] = void_fact

    mu  = fem.Function(chi.function_space)
    lam = fem.Function(chi.function_space)
    mu.x.array[:]  = E/(2*(1+nu))*chi.x.array
    lam.x.array[:] = E*nu/((1+nu)*(1-2*nu))*chi.x.array

    def sigma(w):
        eps = ufl.sym(ufl.grad(w))
        return 2*mu*eps + lam*ufl.tr(eps)*ufl.Identity(2)

    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v)))*ufl.dx
    # body force = 0
    rhs = ufl.Constant(domain, PETSc.ScalarType((0.0, 0.0)))  # dummy

    # 4. Dirichlet BC -----------------------------------------------------
    # uy=0 bottom
    f_bot = mesh.locate_entities_boundary(domain, 1,
              lambda x: np.isclose(x[1], 0.0))
    bc_bot = fem.dirichletbc(PETSc.ScalarType(0.0),
              fem.locate_dofs_topological(V.sub(1), 1, f_bot), V.sub(1))
    # ux=0 bottom-left corner
    v_corner = mesh.locate_entities_boundary(domain, 0,
              lambda x: np.isclose(x[0],0.0)&np.isclose(x[1],0.0))
    bc_corner = fem.dirichletbc(PETSc.ScalarType(0.0),
              fem.locate_dofs_topological(V.sub(0), 0, v_corner), V.sub(0))

    # 5. Neumann load on top edge ----------------------------------------
    f_top = mesh.locate_entities_boundary(
    domain, 1, lambda x: np.isclose(x[1], L))

    bc_top = fem.dirichletbc(
        PETSc.ScalarType(2.0),
        fem.locate_dofs_topological(V.sub(1), 1, f_top),
        V.sub(1))

    zero_rhs = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    problem = petsc.LinearProblem(
        a, ufl.dot(zero_rhs, v)*ufl.dx,
        bcs=[bc_bot, bc_corner, bc_top],
        u=uh,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem.solve()

    # 7. split ------------------------------------------------------------
    V0 = fem.FunctionSpace(domain, ("Lagrange", order))
    ux = fem.Function(V0, name="ux"); ux.x.array[:] = uh.x.array[0::2]
    uy = fem.Function(V0, name="uy"); uy.x.array[:] = uh.x.array[1::2]
    return domain, ux, uy, uh


# ------------------- 生成两种结构 -------------------
mesh_ref,  ux_ref,  uy_ref, u_ref  = solve_case(False)
mesh_hole, ux_hole, uy_hole, u_hole = solve_case(True)

# ------------------- 写入 XDMF ---------------------
def write(path, m, f):
    with io.XDMFFile(m.comm, path, "w") as X:
        X.write_mesh(m)
        X.write_function(f)

write("./data/ux_reference.xdmf", mesh_ref,  ux_ref)
write("./data/uy_reference.xdmf", mesh_ref,  uy_ref)
write("./data/ux_hole.xdmf",     mesh_hole, ux_hole)
write("./data/uy_hole.xdmf",     mesh_hole, uy_hole)
write("./data/u_reference.xdmf",  mesh_ref,  u_ref)
write("./data/u_hole.xdmf",       mesh_hole, u_hole)

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