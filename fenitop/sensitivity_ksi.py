import numpy as np
import torch
import ufl
from mpi4py import MPI
from dolfinx.fem import form
from dolfinx.fem.petsc import create_matrix, assemble_matrix
from petsc4py import PETSc
from fenitop.NN import NeuralNetwork

class Sensitivity_ksi:
    """
    Compute cloak objective Ju and its gradient dJu/dκ
    via adjoint + NN chain rule (no UFL integral for cloak).
    """
    def __init__(self, opt, problem,
                 u_field, lambda_field,
                 ksi_phys_list, c_field_list, local_vf_phys):

        # MPI & FEM objects
        self.comm       = problem.u.function_space.mesh.comm
        self.problem    = problem
        self.u_field    = u_field
        self.lambda_fld = lambda_field

        # cached reference displacement & indicator θ
        self.Uref = opt["U_ref_vec"]    # PETSc.Vec
        self.theta = opt["theta_vec"]   # same layout PETSc.Vec

        # adjoint forms for c_j
        self.dJdc_forms = [
            form(ufl.adjoint(ufl.derivative(opt["f_int"], cj)))
            for cj in c_field_list
        ]
        self.dJdc_mats = [create_matrix(f) for f in self.dJdc_forms]
        # placeholder for (∂K/∂c_j)·U  vectors
        self.dUdc_vecs = [cj.vector.copy() for cj in c_field_list]

        # prepare NN
        m = opt["block_types"] + 1  # frequency hints + vf
        self.nn = NeuralNetwork(input_size=m,
                                hidden1_size=256,
                                hidden2_size=256,
                                output_size=6).double()
        self.nn.load_state_dict(torch.load(
            '/shared/fenitop_for_virtualgrowth/fenitop/trained_model.pth',
            map_location='cpu'))
        self.nn.eval()

        # store κ arrays for _assemble_dcdksi
        self.ksi_arrays = [f.x.array for f in ksi_phys_list] \
                        + [local_vf_phys.x.array for local_vf_phys in [local_vf_phys]]

        # prepare sparse mats ∂c/∂κ
        ne = len(self.ksi_arrays[0])
        self.dcdksi_mats = [
            [PETSc.Mat().createAIJ([ne,ne], nnz=1, comm=self.comm)
             for _ in range(6)]
            for _ in range(m)
        ]
        # preallocate output gradient vectors dJ/dκ
        template = c_field_list[0].vector
        self.dJdksi = [
            [template.copy() for _ in range(6)]
            for _ in range(m)
        ]

    def _assemble_rhs(self, eps=1e-12):
        """
        Compute Ju = ||θ⊙(U-Ū)||₂ / ||θ⊙Ū||₂
        and its derivative wrt U: rhs = ∂Ju/∂U.
        """
        # 1) r = θ ⊙ (U - Ū)
        r = self.u_field.vector.copy()
        r.axpy(-1.0, self.Uref)           # r = U - Ū
        r.pointwiseMult(r, self.theta)    # r = θ ⊙ (U-Ū)

        # 2) numerator squared
        J2 = r.dot(r)                         # ∑ (r_i)²

        # 3) denominator squared = ||θ⊙Ū||²
        tmp = self.Uref.copy()
        tmp.pointwiseMult(tmp, self.theta)  # tmp = θ ⊙ Ū
        R2  = tmp.dot(tmp)                      # ∑ (θ_i Ū_i)²

        # 4) value Ju
        Ju = J2/R2

        # 5) derivative ∂Ju/∂U = (1/R2)*(r/Ju)
        rhs = r.copy()
        rhs.scale(2.0 / R2)
        rhs.pointwiseMult(rhs, self.theta)

        return Ju, rhs

    def _assemble_dJdc(self):
        """
        Solve ∂J/∂c_j = - λᵀ (∂K/∂c_j) U
        using adjoint forms dJdc_forms.
        """
        for mat, form, vec in zip(self.dJdc_mats,
                                  self.dJdc_forms,
                                  self.dUdc_vecs):
            mat.zeroEntries()
            assemble_matrix(mat, form)
            mat.assemble()
            mat.mult(self.lambda_fld.vector, vec)

    def _assemble_dcdksi(self):
        """
        Build sparse ∂c_j/∂κ_i matrices via NN autograd.
        """
        m = len(self.ksi_arrays)
        ne = len(self.ksi_arrays[0])
        for e in range(ne):
            # build input z for element e
            z = np.array([arr[e] for arr in self.ksi_arrays],
                         dtype=np.float64)
            z_t = torch.tensor(z, dtype=torch.float64,
                               requires_grad=True)
            pred = self.nn(z_t)
            for j in range(6):
                grad = torch.autograd.grad(pred[j],
                                           z_t,
                                           retain_graph=True)[0]
                for i in range(m):
                    self.dcdksi_mats[i][j].setValue(e, e,
                                                   grad[i].item())
        # assemble all
        for mats in self.dcdksi_mats:
            for A in mats:
                A.assemble()

    def evaluate(self):
        # 1) assemble Ju and rhs
        Ju, rhs = self._assemble_rhs()
        # 2) solve adjoint Kᵀ λ = rhs
        self.problem.set_adjoint_load(rhs)
        self.problem.solve_adjoint()

        # 3) assemble ∂J/∂c_j
        self._assemble_dJdc()

        # 4) assemble ∂c_j/∂κ_i
        self._assemble_dcdksi()

        # 5) chain rule: dJ/dκ_i = sum_j (∂c_j/∂κ_i)·(dJ/dc_j)
        m = len(self.ksi_arrays)
        for i in range(m):
            for j in range(6):
                out = self.dJdksi[i][j]
                out.zeroEntries()
                self.dcdksi_mats[i][j].mult(self.dUdc_vecs[j], out)
                out.assemble()

        # return only the κ-sensitivity list
        return self.dJdksi
