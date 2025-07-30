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
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags, compute_midpoints
from dolfinx.fem import (VectorFunctionSpace, FunctionSpace, Function, Constant,
                         dirichletbc, locate_dofs_topological, Expression, assemble_scalar, form)
from fenitop.utility import create_mechanism_vectors
from fenitop.utility import LinearProblem
from fenitop.NN import NeuralNetwork
from petsc4py import PETSc
import torch
from dolfinx.io import XDMFFile

from mpi4py import MPI


def form_fem(fem, opt):
    """Form an FEA problem."""
    # Function spaces and functions
    mesh = fem["mesh"]
    mesh_ref = fem["mesh_ref"]
    V = VectorFunctionSpace(mesh, ("CG", 1))
    V_ref = VectorFunctionSpace(mesh_ref, ("CG", 1))
    S0 = FunctionSpace(mesh, ("DG", 0))
    S = FunctionSpace(mesh, ("CG", 1))
    block_types = opt["block_types"]
    max_vf = opt["max_vf"]
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_field = Function(V)  # Displacement field
    lambda_field = Function(V)  # Adjoint variable field
    rho_field = Function(S0)  # Density field
    rho_phys_field = Function(S)  # Physical density field
    ksi_field_list = []  # Frequency hints fields list
    ksi_phys_field_list = []  # Physical frequency hints field list
    c_field_list = []  # Material modulus field list
    local_vf_field = Function(S0)
    local_vf_phys_field = Function(S)

    for i in range(block_types):
        ksi_field_list.append(Function(S0))
        ksi_phys_field_list.append(Function(S))

    for i in range(6):
        c_field_list.append(Function(S))

    if "alpha_field" not in opt:
        V0 = S0
        alpha_field = Function(V0, name="alpha")
        alpha_field.x.array[:] = 0.0

        solid_sel = opt["solid_zone"]
        mids = compute_midpoints(
            mesh, mesh.topology.dim,
            np.arange(mesh.topology.index_map(mesh.topology.dim).size_local,
                      dtype=np.int32))
        inside_solid = solid_sel(mids.T)
        alpha_field.x.array[np.where(inside_solid)[0]] = 1.0
        opt["alpha_field"] = alpha_field
    else:
        alpha_field = opt["alpha_field"]

    # with XDMFFile(fem["mesh"].comm, "/shared/fenitop_for_cloak/data_optimize/alpha_field.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(fem["mesh"]) 
    #     xdmf.write_function(alpha_field) 

    E_mat = fem["young's modulus"]
    nu_mat = fem["poisson's ratio"]
    pref = E_mat / ((1.0 + nu_mat) * (1.0 - 2.0 * nu_mat))

    C11 = pref * (1.0 - nu_mat)
    C12 = pref * nu_mat
    C33 = pref * (1.0 - 2.0 * nu_mat) / 2.0

    C_p_const = [
        Constant(mesh, PETSc.ScalarType(val)) for val in
        [C11, C12, 0.0, C11, 0.0, C33]
    ]

    p_rho, eps_rho = opt["penalty"], opt["epsilon"]
    rho_factor = eps_rho + (1.0 - eps_rho) * rho_phys_field**p_rho

    c_list = []
    for i in range(6):
        Cd_i = c_field_list[i]
        Cp_i = C_p_const[i]
        c_eff = rho_factor * ((1.0 - alpha_field) * Cd_i + alpha_field * Cp_i)
        # c_eff = rho_factor * Cd_i
        c_list.append(c_eff)

    D_matrix = ufl.as_matrix([
        [c_list[0], c_list[1], c_list[2]],
        [c_list[1], c_list[3], c_list[4]],
        [c_list[2], c_list[4], c_list[5]]
    ])

    eps_u = ufl.as_vector([
        u[0].dx(0),           # ε_xx
        u[1].dx(1),           # ε_yy
        u[0].dx(1) + u[1].dx(0)  # engineering shear 2*ε_xy
    ])
    eps_v = ufl.as_vector([
        v[0].dx(0),
        v[1].dx(1),
        v[0].dx(1) + v[1].dx(0)
    ])
    eps_u_field = ufl.as_vector([
        u_field[0].dx(0),
        u_field[1].dx(1),
        u_field[0].dx(1) + u_field[1].dx(0)
    ])

    # S_matrix = ufl.inv(D_matrix)

    # S11 = S_matrix[0, 0]
    # S12 = S_matrix[0, 1]
    # S22 = S_matrix[1, 1]

    # E_field = 0.5*(1/S11+1/S22)
    # nu = -0.5*(S12/S11+S12/S22)

    # _lambda, mu = E_field*nu/(1+nu)/(1-2*nu), E_field/(2*(1+nu))

    # # Kinematics
    # def epsilon(u):
    #     return ufl.sym(ufl.grad(u))

    # def sigma(u):  # 3D or plane strain
    #     return 2*mu*epsilon(u) + _lambda*ufl.tr(epsilon(u))*ufl.Identity(len(u))

    # Boundary conditions
    dim = mesh.topology.dim
    fdim = dim - 1

    f_bot = locate_entities_boundary(mesh, 1,
              lambda x: np.isclose(x[1], -10.0))
    bc_bot = dirichletbc(PETSc.ScalarType(0.0),
              locate_dofs_topological(V.sub(1), 1, f_bot), V.sub(1))
    # ux=0 bottom-left corner
    v_corner = locate_entities_boundary(mesh, 0,
              lambda x: np.isclose(x[0], 0.0)&np.isclose(x[1], -10.0))
    bc_corner = dirichletbc(PETSc.ScalarType(0.0),
              locate_dofs_topological(V.sub(0), 0, v_corner), V.sub(0))
    
    f_top = locate_entities_boundary(mesh, 1,
              lambda x: np.isclose(x[1], 10.0))

    bc_top = dirichletbc(PETSc.ScalarType(2.0),locate_dofs_topological(V.sub(1), 1, f_top),V.sub(1))

    # Combine BCs
    # bcs = [bc_bot, bc_corner, bc_top]
    bcs = [bc_bot, bc_corner, bc_top]
    
    tractions, facets, markers = [], [], []
    for marker, (traction, traction_bc) in enumerate(fem["traction_bcs"]):
        tractions.append(Constant(mesh, np.array(traction, dtype=float)))
        current_facets = locate_entities_boundary(mesh, fdim, traction_bc)
        facets.extend(current_facets)
        markers.extend([marker,]*len(current_facets))
    facets = np.array(facets, dtype=np.int32)
    markers = np.array(markers, dtype=np.int32)
    _, unique_indices = np.unique(facets, return_index=True)
    facets, markers = facets[unique_indices], markers[unique_indices]
    sorted_indices = np.argsort(facets)
    facet_tags = meshtags(mesh, fdim, facets[sorted_indices], markers[sorted_indices])

    metadata = {"quadrature_degree": fem["quadrature_degree"]}
    dx = ufl.Measure("dx", metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags)
    b = Constant(mesh, np.array(fem["body_force"], dtype=float))

    # Establish the equilibrium and adjoint equations
    lhs = ufl.inner(D_matrix*eps_u, eps_v)*dx
    rhs = ufl.dot(b, v)*dx
    for marker, t in enumerate(tractions):
        rhs += ufl.dot(t, v)*ds(marker)
    spring_vec = opt["l_vec"] = None

    linear_problem = LinearProblem(u_field, lambda_field, lhs, rhs, opt["l_vec"],
                                   spring_vec, bcs, fem["petsc_options"])

    # prepare cloaking
    u_ref_field = opt["u_ref_field"]
    u_ref_raw = u_ref_field.vector.copy()
    u_ref_raw.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)

    # 2) 从三分量 array 中抽出 x,y 构造新的 2 分量 array
    raw = u_ref_raw.getArray()           # 长度 = 3 * n_nodes
    n_nodes = raw.size // 3              # 每节点三个自由度
    # 注意 new_arr 长度要正好 = 2 * n_nodes
    new_arr = np.empty(2 * n_nodes, dtype=raw.dtype)
    new_arr[0::2] = raw[0::3]            # x 分量
    new_arr[1::2] = raw[1::3]            # y 分量

    U_ref = Function(V_ref)
    U_ref_vec = U_ref.vector.copy()  # PETSc.Vec template

    # 3) 用 u_field.vector (2 分量) 作为模板创建新的参考 Vec
    U_ref_vec.setArray(new_arr)          # 直接把 array 贴上
    U_ref_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)

    # # 4) 缓存

    # # 5) 计算theta

    # cell_dim = mesh_ref.topology.dim
    # cells_ref = np.arange(mesh_ref.topology.index_map(cell_dim).size_local, dtype=np.int32)

    # # mid‑points of reference and perforated cells (local arrays)
    # mids_ref = compute_midpoints(mesh_ref,  cell_dim, cells_ref)
    # mid_set_hole = {
    #     tuple(np.round(pt, 8))
    #     for pt in compute_midpoints(
    #         mesh,
    #         cell_dim,
    #         np.arange(mesh.topology.index_map(cell_dim).size_local, dtype=np.int32),
    #     )
    # }

    # V0_nodal = FunctionSpace(mesh_ref, ("CG", 1))
    # mask_scalar = Function(V0_nodal, name="mask_scalar")
    # mask_scalar.x.array[:] = 1.0        # 默认全部置 1

    # # 将属于“hole cell”的所有顶点自由度置 0
    # dofmap = V0_nodal.dofmap
    # for cell_lid, pt in enumerate(mids_ref):
    #     if tuple(np.round(pt, 8)) not in mid_set_hole:
    #         mask_scalar.x.array[dofmap.cell_dofs(cell_lid)] = 0.0
    # mask_scalar.x.scatter_forward()     # 同步幽灵自由度 (并行安全)

    # # 2.2 扩展成 **vector nodal** mask，使长度 = 位移向量
    # theta = Function(V_ref)

    # vals = mask_scalar.x.array
    # theta.x.array[0::2] = vals         # x-分量
    # theta.x.array[1::2] = vals         # y-分量
    # theta.x.scatter_forward()
    # theta_vec = theta.vector.copy()
    # opt["theta_vec"] = theta_vec    # 与位移 Vec 尺寸完全一致

    # # 6) 投影u到与u_ref相同大小的函数空间
    # V0 = FunctionSpace(mesh, ("Lagrange", 1))
    # ux = Function(V0, name="ux")
    # uy = Function(V0, name="uy")
    # ux.x.array[:] = u_field.x.array[0::2]
    # uy.x.array[:] = u_field.x.array[1::2]
    # V0_ref = FunctionSpace(mesh_ref, ("Lagrange", 1))
    # V0_hole = ux.function_space  # scalar Lagrange space on the perforated mesh

    # coords_ref = V0_ref.tabulate_dof_coordinates()
    # coords_hole = V0_hole.tabulate_dof_coordinates()

    # ux_interp = Function(V0_ref, name="ux_hole_interp")
    # uy_interp = Function(V0_ref, name="uy_hole_interp")

    # # build look‑up tables  {rounded‑coord: value}
    # key = lambda xy: (round(float(xy[0]), 8), round(float(xy[1]), 8))
    # ux_map = {key(xy): val for xy, val in zip(coords_hole, ux.x.array)}
    # uy_map = {key(xy): val for xy, val in zip(coords_hole, uy.x.array)}

    # ux_vals = ux_interp.x.array  # local view (to be filled)
    # uy_vals = uy_interp.x.array

    # for i, xy in enumerate(coords_ref):
    #     k = key(xy)
    #     ux_vals[i] = ux_map.get(k, 0.0)
    #     uy_vals[i] = uy_map.get(k, 0.0)

    # u_interp = Function(V_ref, name="u_hole_interp_vec")
    # vec_arr = u_interp.x.array
    # vec_arr[0::2] = ux_vals
    # vec_arr[1::2] = uy_vals
    # u_interp.x.scatter_forward()

    # opt["u_interp"] = u_interp

    V0_ref = FunctionSpace(mesh_ref, ("Lagrange", 1))

    ux_ref = Function(V0_ref, name="ux_ref")
    uy_ref = Function(V0_ref, name="uy_ref")
    ux_ref.x.array[:] = U_ref_vec.getArray()[0::2]
    uy_ref.x.array[:] = U_ref_vec.getArray()[1::2]
    ux_ref.x.scatter_forward()
    uy_ref.x.scatter_forward()

    # ---------------------------------------------------------------
    # ❷ 创建“目标”函数空间：带孔网格上的 P¹ 向量场
    # ---------------------------------------------------------------
    V0_hole = FunctionSpace(mesh, ("Lagrange", 1))           # scalar

    # ---------------------------------------------------------------
    # ❸ 用坐标-字典把 (ux_ref, uy_ref) 复制到带孔顶点
    # ---------------------------------------------------------------
    coords_ref = V0_ref.tabulate_dof_coordinates()
    coords_hole = V0_hole.tabulate_dof_coordinates()

    key = lambda xy: (round(float(xy[0]), 8), round(float(xy[1]), 8))
    ux_map = {key(xy): val for xy, val in zip(coords_ref, ux_ref.x.array)}
    uy_map = {key(xy): val for xy, val in zip(coords_ref, uy_ref.x.array)}

    # scalar fields on *hole* mesh
    ux_ref_hole = Function(V0_hole, name="ux_ref_on_hole")
    uy_ref_hole = Function(V0_hole, name="uy_ref_on_hole")

    for i, xy in enumerate(coords_hole):
        k = key(xy)
        ux_ref_hole.x.array[i] = ux_map.get(k, 0.0)   # fill 0 if vertex not in ref mesh
        uy_ref_hole.x.array[i] = uy_map.get(k, 0.0)

    ux_ref_hole.x.scatter_forward()
    uy_ref_hole.x.scatter_forward()

    # ---------------------------------------------------------------
    # ❹ 重新打包成向量 Function（与 u_field 同空间、同长度）
    # ---------------------------------------------------------------
    U_ref_hole = Function(V, name="U_ref_on_hole")
    arr = U_ref_hole.x.array          # writable view
    arr[0::2] = ux_ref_hole.x.array
    arr[1::2] = uy_ref_hole.x.array
    U_ref_hole.x.scatter_forward()

    opt["U_ref_vec"] = U_ref_hole.vector.copy()

    # Define optimization-related variables
    opt["f_int"] = ufl.inner(D_matrix*eps_u_field, eps_v)*dx
    opt["compliance"] = ufl.inner(D_matrix*eps_u_field, eps_u_field)*dx

    opt["volume"] = rho_phys_field * local_vf_phys_field * dx
    opt["total_volume"] = Constant(mesh, 1.0 * max_vf) * dx

    return linear_problem, u_field, lambda_field, rho_field, rho_phys_field, ksi_field_list, ksi_phys_field_list, c_field_list, local_vf_field, local_vf_phys_field
