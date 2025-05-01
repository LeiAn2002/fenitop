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

import ufl
from mpi4py import MPI
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import create_vector, create_matrix, assemble_vector, assemble_matrix
from petsc4py import PETSc


class Sensitivity():
    def __init__(self, comm, opt, problem, u_field, lambda_field, rho_phys, local_vf_phys_field):
        # Compliance
        # if opt["opt_compliance"]:
        #     self.C_form = form(opt["compliance"])
        # self.dCdrho_form = form(-ufl.derivative(opt["compliance"], rho_phys))
        # self.dCdrho_vec = create_vector(self.dCdrho_form)

        # Volume
        # print(max(rho_phys.vector.array))
        self.comm = comm
        self.problem = problem
        self.u_field = u_field
        self.lambda_field = lambda_field

        self.J_form = form(opt["cloak"])
        self.dJdU_form = form(ufl.derivative(opt["cloak"], u_field))
        self.dJdU_vec = create_vector(self.dJdU_form)

        self.dJdrho_form = form(ufl.adjoint(
            ufl.derivative(opt["f_int"], rho_phys)))
        self.dJdrho_mat = create_matrix(self.dJdrho_form)
        self.dJdrho_vec = rho_phys.vector.copy()

        self.total_volume = comm.allreduce(
            assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)
        self.V_form = form(opt["volume"])
        # self.dVdrho_form = form(ufl.derivative(opt["volume"], rho_phys))
        # self.dVdrho_vec = create_vector(self.dVdrho_form)

        self.dVdvf_form = form(ufl.derivative(opt["volume"], local_vf_phys_field))
        self.dVdvf_vec = create_vector(self.dVdvf_form)

        # Displacement
        # self.opt_compliance = opt["opt_compliance"]
        # if not self.opt_compliance:
        #     self.dfdrho_form = form(ufl.adjoint(ufl.derivative(opt["f_int"], rho_phys)))

        #     self.dfdrho_mat = create_matrix(self.dfdrho_form)

        #     self.problem, self.l_vec = problem, opt["l_vec"]
        #     self.u_field, self.lambda_field = u_field, lambda_field
        #     self.dUdrho_vec = rho_phys.vector.copy()
        #     self.prod_vec = u_field.vector.copy()

    def evaluate(self):
        # Compliance
        # if self.opt_compliance:
        #     C_value = self.comm.allreduce(assemble_scalar(self.C_form), op=MPI.SUM)
        # else:
        #     self.problem.lhs_mat.mult(self.u_field.vector, self.prod_vec)
        #     C_value = self.u_field.vector.dot(self.prod_vec)
        # with self.dCdrho_vec.localForm() as loc:
        #     loc.set(0)
        # assemble_vector(self.dCdrho_vec, self.dCdrho_form)
        # self.dCdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        J_value = self.comm.allreduce(
            assemble_scalar(self.J_form), op=MPI.SUM)

        with self.dJdU_vec.localForm() as loc:
            loc.set(0)
        assemble_vector(self.dJdU_vec, self.dJdU_form)
        self.dJdU_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                  mode=PETSc.ScatterMode.REVERSE)

        # solve Kᵀ λ = ∂J/∂U
        self.problem.set_adjoint_load(self.dJdU_vec)
        self.problem.solve_adjoint()

        # ---- assemble ∂K/∂ρ and multiply ----
        self.dJdrho_mat.zeroEntries()
        assemble_matrix(self.dJdrho_mat, self.dJdrho_form)
        self.dJdrho_mat.assemble()
        self.dJdrho_mat.mult(self.lambda_field.vector, self.dJdrho_vec)

        # assemble_vector(self.dVdrho_vec, self.dVdrho_form)
        # self.dVdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        # self.dVdrho_vec /= self.total_volume

        actual_volume = self.comm.allreduce(assemble_scalar(self.V_form), op=MPI.SUM)
        V_value = actual_volume / self.total_volume

        assemble_vector(self.dVdvf_vec, self.dVdvf_form)
        self.dVdvf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.dVdvf_vec /= self.total_volume

        # Displacement
        # if not self.opt_compliance:
        #     U_value = self.u_field.vector.dot(self.l_vec)
        #     self.problem.solve_adjoint()
        #     self.dfdrho_mat.zeroEntries()
        #     assemble_matrix(self.dfdrho_mat, self.dfdrho_form)
        #     self.dfdrho_mat.assemble()
        #     self.dfdrho_mat.mult(self.lambda_field.vector, self.dUdrho_vec)
        # else:
        #     U_value, self.dUdrho_vec = 0, None

        func_value = [J_value, V_value]
        sensitivity = [self.dJdrho_vec]
        return func_value, sensitivity, self.dVdvf_vec
