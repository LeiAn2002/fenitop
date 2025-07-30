# -*- coding: utf-8 -*-
"""
Pre-processing for mechanical cloak — displacement‑controlled version
This script reproduces the two reference cases (solid plate / plate‑with‑hole),
except that the perforated geometry is now provided directly as a Gmsh *.msh*
file.  All boundary conditions and material data follow the original script.

* author: ChatGPT
* date  : 2025‑07‑29
"""

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem import petsc

# --------------------------- user parameters ---------------------------
L = 20.0                # plate length  (mm)
nel = 100                # element divisions for the reference square mesh
order = 1               # polynomial order of Lagrange elements
E, nu = 2000, 0.34      # Young's modulus & Poisson's ratio of the solid

top_disp = 2.0          # prescribed displacement on the top edge (mm)

# path to the gmsh mesh containing the irregular hole
msh_file = "./meshes/eye.msh"


# --------------------------- helper routines ---------------------------

def build_domain(with_hole: bool) -> mesh.Mesh:
    """Create a *dolfinx* mesh.

    If *with_hole* is *True* the geometry is imported from *msh_file*;
    otherwise a structured quadrilateral mesh is generated programmatically.
    """
    if with_hole:
        # read_from_msh returns (mesh, cell_tags, facet_tags)
        domain, _, _ = io.gmshio.read_from_msh(
            msh_file, comm=MPI.COMM_WORLD, rank=0, gdim=2
        )
    else:
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [[-L / 2, -L / 2], [L / 2, L / 2]],
            [nel, nel],
            cell_type=mesh.CellType.quadrilateral,
        )
    return domain


def solve_case(with_hole: bool):
    """Solve the small‑strain elasticity problem for the given geometry."""
    domain = build_domain(with_hole)

    # --- function spaces ------------------------------------------------
    V = fem.FunctionSpace(
        domain, ufl.VectorElement("Lagrange", domain.ufl_cell(), order, dim=2)
    )
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    uh = fem.Function(V, name="u")

    # --- linear elastic material (isotropic) ---------------------------
    mu_val = E / (2 * (1 + nu))
    lam_val = E * nu / ((1 + nu) * (1 - 2 * nu))

    mu = fem.Constant(domain, PETSc.ScalarType(mu_val))
    lam = fem.Constant(domain, PETSc.ScalarType(lam_val))

    def sigma(w):
        eps = ufl.sym(ufl.grad(w))
        return 2 * mu * eps + lam * ufl.tr(eps) * ufl.Identity(2)

    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    rhs = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))  # no body force

    # --- Dirichlet boundary conditions ---------------------------------
    tol = 1.0e-8

    # uy = 0  on the bottom edge
    f_bot = mesh.locate_entities_boundary(
        domain,
        domain.topology.dim - 1,
        lambda x: np.isclose(x[1], -L / 2, atol=tol),
    )
    bc_bot = fem.dirichletbc(
        PETSc.ScalarType(0.0),
        fem.locate_dofs_topological(V.sub(1), domain.topology.dim - 1, f_bot),
        V.sub(1),
    )

    # ux = 0  at the bottom‑left corner (eliminates rigid‑body motion)
    v_corner = mesh.locate_entities_boundary(
        domain,
        0,
        lambda x: np.isclose(x[0], -L / 2, atol=tol)
        & np.isclose(x[1], -L / 2, atol=tol),
    )
    bc_corner = fem.dirichletbc(
        PETSc.ScalarType(0.0),
        fem.locate_dofs_topological(V.sub(0), 0, v_corner),
        V.sub(0),
    )

    # uy = +top_disp  on the top edge
    f_top = mesh.locate_entities_boundary(
        domain,
        domain.topology.dim - 1,
        lambda x: np.isclose(x[1], L / 2, atol=tol),
    )
    bc_top = fem.dirichletbc(
        PETSc.ScalarType(top_disp),
        fem.locate_dofs_topological(V.sub(1), domain.topology.dim - 1, f_top),
        V.sub(1),
    )

    problem = petsc.LinearProblem(
        a,
        ufl.dot(rhs, v) * ufl.dx,
        bcs=[bc_bot, bc_corner, bc_top],
        u=uh,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    problem.solve()

    # --- split displacement into scalar fields for post‑processing ------
    V0 = fem.FunctionSpace(domain, ("Lagrange", order))
    ux = fem.Function(V0, name="ux")
    uy = fem.Function(V0, name="uy")
    ux.x.array[:] = uh.x.array[0::2]
    uy.x.array[:] = uh.x.array[1::2]

    return domain, ux, uy, uh


# --------------------------- main workflow -----------------------------
mesh_ref, ux_ref, uy_ref, u_ref = solve_case(False)
mesh_irr, ux_irr, uy_irr, u_irr = solve_case(True)


# --------------------------- XDMF export -------------------------------

def write(path: str, m: mesh.Mesh, f: fem.Function):
    """Write *mesh* and *function* to an XDMF file."""
    with io.XDMFFile(m.comm, path, "w") as X:
        X.write_mesh(m)
        X.write_function(f)


outputs = [
    ("./data/ux_reference.xdmf", mesh_ref, ux_ref),
    ("./data/uy_reference.xdmf", mesh_ref, uy_ref),
    ("./data/ux_hole.xdmf", mesh_irr, ux_irr),
    ("./data/uy_hole.xdmf", mesh_irr, uy_irr),
    ("./data/u_reference.xdmf", mesh_ref, u_ref),
    ("./data/u_hole.xdmf", mesh_irr, u_irr),
]
for arg in outputs:
    write(*arg)

if mesh_ref.comm.rank == 0:
    print("✅ displacement fields saved — open the .xdmf files in ParaView.")

# --------------------------- error analysis -------------------------------

cell_dim = mesh_ref.topology.dim
cells_ref = np.arange(mesh_ref.topology.index_map(cell_dim).size_local, dtype=np.int32)

# mid‑points of reference and perforated cells (local arrays)
mids_ref  = mesh.compute_midpoints(mesh_ref,  cell_dim, cells_ref)
mid_set_hole = {
    tuple(np.round(pt, 8))
    for pt in mesh.compute_midpoints(
        mesh_irr,
        cell_dim,
        np.arange(mesh_irr.topology.index_map(cell_dim).size_local, dtype=np.int32),
    )
}

V_mask = fem.FunctionSpace(mesh_ref, ("DG", 0))
mask = fem.Function(V_mask, name="mask")
mask_array = np.ones_like(mask.x.array, dtype=mask.x.array.dtype)

for local_id, pt in enumerate(mids_ref):
    if tuple(np.round(pt, 8)) not in mid_set_hole:
        mask_array[local_id] = 0.0  # this cell belongs to the hole
mask.x.array[:] = mask_array

# --- 4.2  interpolate hole‑solution onto reference vertices -------------
V0_ref  = fem.FunctionSpace(mesh_ref, ("Lagrange", order))
V0_hole = ux_irr.function_space  # scalar Lagrange space on the perforated mesh

coords_ref  = V0_ref.tabulate_dof_coordinates()
coords_hole = V0_hole.tabulate_dof_coordinates()

ux_interp = fem.Function(V0_ref, name="ux_hole_interp")
uy_interp = fem.Function(V0_ref, name="uy_hole_interp")

# build look‑up tables  {rounded‑coord: value}
key = lambda xy: (round(float(xy[0]), 8), round(float(xy[1]), 8))
ux_map = {key(xy): val for xy, val in zip(coords_hole, ux_irr.x.array)}
uy_map = {key(xy): val for xy, val in zip(coords_hole, uy_irr.x.array)}

ux_vals = ux_interp.x.array  # local view (to be filled)
uy_vals = uy_interp.x.array

for i, xy in enumerate(coords_ref):
    k = key(xy)
    ux_vals[i] = ux_map.get(k, 0.0)
    uy_vals[i] = uy_map.get(k, 0.0)

# --- 4.3  assemble masked L²‑error ------------------------------------

dx_ref = ufl.dx(domain=mesh_ref)

def l2_sq_masked(f):
    local = fem.assemble_scalar(fem.form(f * f * mask * dx_ref))
    return mesh_ref.comm.allreduce(local, op=MPI.SUM)

num_x = l2_sq_masked(ux_interp - ux_ref)
num_y = l2_sq_masked(uy_interp - uy_ref)

den_x = l2_sq_masked(ux_ref)
den_y = l2_sq_masked(uy_ref)

ex = np.sqrt(num_x) / np.sqrt(den_x) if den_x > 0 else float("nan")
ey = np.sqrt(num_y) / np.sqrt(den_y) if den_y > 0 else float("nan")

if mesh_ref.comm.rank == 0:
    print(
        f"🔎  L²‑relative errors (void excluded):  ex = {ex*100:.2f} %,   ey = {ey*100:.2f} %"
    )
