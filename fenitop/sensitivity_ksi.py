import ufl
from mpi4py import MPI
from dolfinx.fem import form, assemble_scalar, Function, FunctionSpace, petsc
from dolfinx.fem.petsc import create_vector, create_matrix, assemble_vector, assemble_matrix
from petsc4py import PETSc
from fenitop.NN import NeuralNetwork
import numpy as np

import torch

class Sensitivity_ksi():
    """
    Compute cloak objective J and its gradient dJ/dκ  (κ = frequency hints + vf).
    No compliance / mechanism remnants inside.
    """
    def __init__(self, opt, problem, u_field, lambda_field,
                 ksi_phys_field_list, c_field_list, local_vf_phys_field):

        self.comm       = problem.u.function_space.mesh.comm
        self.problem    = problem
        self.u_field    = u_field
        self.lambda_fld = lambda_field
        self.opt        = opt
        # ------- objective & derivatives wrt U ------------------
        # self.J_form  = form(opt["J"])
        self.dJdU_vec = create_vector(form(
            ufl.derivative(opt["cloak"], u_field)))

        # adjoint matrix list  dJ/dc_j = -λᵀ ∂K/∂c_j U ------------
        self.dJdc_form  = [form(ufl.adjoint(
                            ufl.derivative(opt["f_int"], cj)))
                           for cj in c_field_list]
        self.dJdc_mat   = [create_matrix(f) for f in self.dJdc_form]
        self.dUdc_vec   = [cj.vector.copy() for cj in c_field_list]

        # ------- NN Jacobian containers -------------------------
        m = opt["block_types"] + 1     # last dim = vf
        nelem = len(local_vf_phys_field.x.array)
        self.dcdksi_mat = [[PETSc.Mat().createAIJ([nelem, nelem], nnz=1,
                                                  comm=self.comm)
                            for _ in range(6)] for _ in range(m)]
        self.dJdksi_vec = [[c_field_list[0].vector.copy() for _ in range(6)] for _ in range(m)]

        # store κ arrays & NN
        self.ksi_arrays = [f.x.array for f in ksi_phys_field_list] \
                        + [local_vf_phys_field.x.array]
        
        self.nn = NeuralNetwork(input_size=m, hidden1_size=256, hidden2_size=256, output_size=6)
        self.nn.load_state_dict(torch.load('/shared/fenitop_for_virtualgrowth/fenitop/trained_model.pth'))
        self.nn.eval()
        self.nn = self.nn.double()

    # ------------------------------------------------------------
    def _assemble_dJdU(self):
        with self.dJdU_vec.localForm() as loc:
            loc.set(0)
        assemble_vector(self.dJdU_vec,
                        form(ufl.derivative(self.opt["cloak"], self.u_field)))
        self.dJdU_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                  mode=PETSc.ScatterMode.REVERSE)

    def _solve_adjoint(self):
        self.problem.set_adjoint_load(self.dJdU_vec)
        self.problem.solve_adjoint()

    def _assemble_dJdc(self):
        for mat, f, vec in zip(self.dJdc_mat, self.dJdc_form, self.dUdc_vec):
            mat.zeroEntries()
            assemble_matrix(mat, f)
            mat.assemble()
            mat.mult(self.lambda_fld.vector, vec)

    def _assemble_dcdksi(self):
        m = len(self.ksi_arrays)
        ne = len(self.ksi_arrays[0])
        for e in range(ne):
            z = np.array([arr[e] for arr in self.ksi_arrays], dtype=np.float64)
            z_t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
            pred = self.nn(z_t)
            for j in range(6):
                grad = torch.autograd.grad(pred[j], z_t, retain_graph=True)[0]
                for i in range(m):
                    self.dcdksi_mat[i][j].setValue(e, e, grad[i].item())
        for mats in self.dcdksi_mat:
            for A in mats:
                A.assemble()

    # ------------------------------------------------------------
    def evaluate(self):
        # (1) 目标值
        # J_val = self.comm.allreduce(
        #     assemble_scalar(self.J_form), op=MPI.SUM)

        # (2) dJ/dU 向量 & adjoint
        self._assemble_dJdU()
        self._solve_adjoint()

        # (3) dJ/dc_j via λ
        self._assemble_dJdc()

        # (4) ∂c/∂κ via NN
        self._assemble_dcdksi()

        # (5) chain rule  dJ/dκ = dJ/dc · ∂c/∂κ
        m = len(self.ksi_arrays)
        for i in range(m):
            for j in range(6):
                vec = self.dJdksi_vec[i][j]
                vec.zeroEntries()
                self.dcdksi_mat[i][j].mult(self.dUdc_vec[j], vec)
                vec.assemble()

        return self.dJdksi_vec