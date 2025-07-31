# -*- coding: utf-8 -*-
"""
Post‑processing script
======================

Evaluates the L²‑relative displacement error between
  • *u_opt*  – displacement of the **optimised perforated plate**
  • *u_ref*  – displacement of the **solid reference plate**

Key points
-----------
* The **reference field lives on a full square mesh** (no hole), while the
  optimised field lives on a **Gmsh mesh with an irregular hole**.
* We therefore **interpolate u_opt onto the reference mesh**, filling the hole
  vertices with zeros, and build a DG‑0 *mask* to exclude hole cells when
  integrating.

Assumptions
-----------
* Square plate side length   L = 50
* Structured reference mesh  nel × nel  with nel = 50
* Gmsh file of perforated mesh at   ./mesh/plate_with_hole.msh
* Displacement fields saved in HDF5 (fenitop.utility.load_field_from_h5)
    − Solid   reference :  ./data/u_reference.h5
    − Optimised result :  ./data_optimize/u_field.h5

Run with:  `python evaluate_irregular_hole_error.py`
"""

import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.mesh import create_rectangle, CellType, compute_midpoints
from dolfinx.io import gmshio
import ufl

from fenitop.utility import load_field_from_h5

# ------------------------------------------------------------------
# 0. user parameters
# ------------------------------------------------------------------
L   = 20.0
nel = 100
msh_file = "./meshes/eye.msh"
path_u_opt = "./data_optimize/u_field.h5"
path_u_ref = "./data/u_reference.h5"

# ------------------------------------------------------------------
# 1. build meshes: solid reference & perforated optimised
# ------------------------------------------------------------------
mesh_ref = create_rectangle(MPI.COMM_WORLD,
                            [[-L/2, -L/2], [L/2, L/2]],
                            [nel, nel],
                            CellType.quadrilateral)
mesh_hole, _, _ = gmshio.read_from_msh(
    msh_file, comm=MPI.COMM_WORLD, rank=0, gdim=2)

# ------------------------------------------------------------------
# 2. read displacement fields (PETSc.Vec)
# ------------------------------------------------------------------
u_opt, _ = load_field_from_h5(mesh_hole, path_u_opt)   # vector on perforated mesh
u_ref, _ = load_field_from_h5(mesh_ref,  path_u_ref)   # vector on solid mesh

u_opt.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)
u_ref.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)

# ------------------------------------------------------------------
# 3. extract scalar components of u_opt on the hole mesh
# ------------------------------------------------------------------
order = 1
V0_hole = fem.FunctionSpace(mesh_hole, ("Lagrange", order))
ux_opt = fem.Function(V0_hole, name="ux_opt")
uy_opt = fem.Function(V0_hole, name="uy_opt")
ux_opt = u_opt.sub(0).collapse()
uy_opt = u_opt.sub(1).collapse()
# ux_opt.x.scatter_forward(); uy_opt.x.scatter_forward()

# ------------------------------------------------------------------
# 4. interpolate (copy) u_opt to reference mesh
# ------------------------------------------------------------------
V0_ref  = fem.FunctionSpace(mesh_ref, ("Lagrange", order))

coords_ref  = V0_ref.tabulate_dof_coordinates()
coords_hole = V0_hole.tabulate_dof_coordinates()

# build {coord: value} look‑up tables
key = lambda xy: (round(float(xy[0]), 8), round(float(xy[1]), 8))
ux_map = {key(pt): val for pt, val in zip(coords_hole, ux_opt.x.array)}
uy_map = {key(pt): val for pt, val in zip(coords_hole, uy_opt.x.array)}

ux_interp = fem.Function(V0_ref, name="ux_opt_interp")
uy_interp = fem.Function(V0_ref, name="uy_opt_interp")

for i, xy in enumerate(coords_ref):
    k = key(xy)
    ux_interp.x.array[i] = ux_map.get(k, 0.0)   # 0.0 if vertex lies in hole
    uy_interp.x.array[i] = uy_map.get(k, 0.0)
ux_interp.x.scatter_forward(); uy_interp.x.scatter_forward()

# repack into a vector Function on mesh_ref
V_vec_ref = fem.FunctionSpace(mesh_ref,
    ufl.VectorElement("Lagrange", mesh_ref.ufl_cell(), order, dim=2))

u_opt_ref = fem.Function(V_vec_ref, name="u_opt_on_ref")
arr = u_opt_ref.vector.array
arr[0::2] = ux_interp.x.array
arr[1::2] = uy_interp.x.array
u_opt_ref.x.scatter_forward()

# ------------------------------------------------------------------
# 5. build DG‑0 mask = 1 (solid) / 0 (hole) on reference mesh
# ------------------------------------------------------------------
cell_dim = mesh_ref.topology.dim
cells_ref = np.arange(mesh_ref.topology.index_map(cell_dim).size_local, dtype=np.int32)

mids_ref  = compute_midpoints(mesh_ref,  cell_dim, cells_ref)
mid_set_hole = {
    tuple(np.round(pt, 8))
    for pt in compute_midpoints(
        mesh_hole,
        cell_dim,
        np.arange(mesh_hole.topology.index_map(cell_dim).size_local, dtype=np.int32))
}

V_mask = fem.FunctionSpace(mesh_ref, ("DG", 0))
mask = fem.Function(V_mask, name="mask")
mask.x.array[:] = 1.0
for cid, pt in enumerate(mids_ref):
    if tuple(np.round(pt, 8)) not in mid_set_hole:
        mask.x.array[cid] = 0.0   # this cell lies in the hole
mask.x.scatter_forward()

# ------------------------------------------------------------------
# 6. compute masked L²‑relative error
# ------------------------------------------------------------------

dx_ref = ufl.dx(domain=mesh_ref)


def l2_sq(f):
    local = fem.assemble_scalar(fem.form(f * f * mask * dx_ref))
    return mesh_ref.comm.allreduce(local, op=MPI.SUM)

ux_ref = u_ref.sub(0).collapse()
uy_ref = u_ref.sub(1).collapse()
ux_opt_on_ref = u_opt_ref.sub(0).collapse()
uy_opt_on_ref = u_opt_ref.sub(1).collapse()

num_x = l2_sq(ux_opt_on_ref - ux_ref)
num_y = l2_sq(uy_opt_on_ref - uy_ref)

den_x = l2_sq(ux_ref)

den_y = l2_sq(uy_ref)

ex = math.sqrt(num_x) / math.sqrt(den_x) if den_x > 0 else float("nan")
ey = math.sqrt(num_y) / math.sqrt(den_y) if den_y > 0 else float("nan")

if mesh_ref.comm.rank == 0:
    print(f"🔎  L²‑relative errors (void excluded):  ex = {ex*100:.2f} %,   ey = {ey*100:.2f} %\n")
