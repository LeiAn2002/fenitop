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


def form_fem(fem, opt):
    """Form an FEA problem."""
    # Function spaces and functions
    mesh = fem["mesh"]
    V = VectorFunctionSpace(mesh, ("CG", 1))
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
        c_list.append(c_eff)

    D_matrix = ufl.as_matrix([
        [c_list[0], c_list[1], c_list[2]],
        [c_list[1], c_list[3], c_list[4]],
        [c_list[2], c_list[4], c_list[5]]
    ])

    S_matrix = ufl.inv(D_matrix)

    S11 = S_matrix[0, 0]
    S12 = S_matrix[0, 1]
    S22 = S_matrix[1, 1]

    E_field = 0.5*(1/S11+1/S22)
    nu = -0.5*(S12/S11+S12/S22)

    _lambda, mu = E_field*nu/(1+nu)/(1-2*nu), E_field/(2*(1+nu))

    # Kinematics
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):  # 3D or plane strain
        return 2*mu*epsilon(u) + _lambda*ufl.tr(epsilon(u))*ufl.Identity(len(u))

    # Boundary conditions
    dim = mesh.topology.dim
    fdim = dim - 1
    dirichlet_bcs = []

    for val_vec, locator, comps in fem["disp_bcs"]:
        const_val = Constant(mesh, np.array(val_vec, dtype=float))
        target_facets = locate_entities_boundary(mesh, fdim, locator)

        if comps is None:  # apply to full vector
            dofs = locate_dofs_topological(V, fdim, target_facets)
            dirichlet_bcs.append(dirichletbc(const_val, dofs, V))
        else:              # apply to selected components
            for c in comps:                          # c = 0 (ux) or 1 (uy)
                dofs_c = locate_dofs_topological(V.sub(c), fdim, target_facets)
                dirichlet_bcs.append(
                    dirichletbc(PETSc.ScalarType(val_vec[c]), dofs_c, V.sub(c)))

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
    lhs = ufl.inner(sigma(u), epsilon(v))*dx
    rhs = ufl.dot(b, v)*dx
    for marker, t in enumerate(tractions):
        rhs += ufl.dot(t, v)*ds(marker)
    if opt["opt_compliance"]:
        spring_vec = opt["l_vec"] = None
    else:
        spring_vec, opt["l_vec"] = create_mechanism_vectors(
            V, opt["in_spring"], opt["out_spring"])

    linear_problem = LinearProblem(u_field, lambda_field, lhs, rhs, opt["l_vec"],
                                   spring_vec, dirichlet_bcs, fem["petsc_options"])

    # prepare cloaking
    if "mask_control" not in opt:
        V0 = S0
        mask_control = Function(V0, name="mask_control")
        mask_control.x.array[:] = 1.0

        void_selector = opt["void_zone"]

        mids = compute_midpoints(
            mesh, mesh.topology.dim,
            np.arange(mesh.topology.index_map(mesh.topology.dim).size_local,
                      dtype=np.int32))

        inside_bool = void_selector(mids.T)
        mask_control.x.array[np.where(inside_bool)[0]] = 0.0

        opt["mask_control"] = mask_control
    else:
        mask_control = opt["mask_control"]

    u_ref_field = opt["u_ref_field"]
    eps_ref = 1e-12
    theta_u = Constant(mesh, np.array((1.0, 1.0)))

    rel_vec = theta_u * (u_field - u_ref_field) / (u_ref_field + eps_ref)
    err_density = ufl.inner(rel_vec, rel_vec) * mask_control

    # Define optimization-related variables
    opt["f_int"] = ufl.inner(sigma(u_field), epsilon(v))*dx
    opt["compliance"] = ufl.inner(sigma(u_field), epsilon(u_field))*dx
    opt["cloak"] = err_density * dx

    opt["volume"] = local_vf_phys_field * dx
    opt["total_volume"] = Constant(mesh, 1.0 * max_vf) * dx

    return linear_problem, u_field, lambda_field, rho_field, rho_phys_field, ksi_field_list, ksi_phys_field_list, c_field_list, local_vf_field, local_vf_phys_field
