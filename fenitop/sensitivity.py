import numpy as np, ufl
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import create_vector, create_matrix, assemble_matrix, assemble_vector
from petsc4py import PETSc
from mpi4py import MPI

class Sensitivity:
    """
    Value & gradient of Ju (cloak) + volume
    """
    def __init__(self, comm, opt, problem,
                 u_field, lambda_field, rho_phys, local_vf_phys):
        self.comm, self.problem = comm, problem
        self.u, self.lam = u_field, lambda_field
        self.theta_vec   = opt["theta_vec"]
        self.Uref_vec    = opt["U_ref_vec"]

        # Ju 对 ρ 的链式项  -λᵀ ∂K/∂ρ U
        self.dJdrho_form = form(ufl.adjoint(
            ufl.derivative(opt["f_int"], rho_phys)))
        self.dJdrho_mat  = create_matrix(self.dJdrho_form)
        self.dJdrho_vec  = rho_phys.vector.copy()

        # volume（约束）
        self.total_vol = comm.allreduce(
            assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)
        self.V_form     = form(opt["volume"])
        self.dVdvf_form = form(ufl.derivative(opt["volume"], local_vf_phys))
        self.dVdvf_vec = create_vector(self.dVdvf_form)

    # ---------- Ju 及伴随右端 ----------
    def _assemble_J_and_rhs(self):
        """
        Compute Ju = ||θ⊙(U-Ū)||₂ / ||θ⊙Ū||₂
        and its derivative wrt U: ∂Ju/∂U = θ⊙(U-Ū) / (Ju * ||θ⊙Ū||₂²).
        All in PETSc.Vec operations, O(n_dofs).
        """
        # (1) r = θ ⊙ (U - Ū)
        r = self.u.vector.copy()
        r.axpy(-1.0, self.Uref_vec)      # r = U - Ū
        r.pointwiseMult(r, self.theta_vec)

        # (2) compute numerator J2 = rᵀ r
        J2 = r.dot(r)

        # (3) compute global denom R2 = (θ⊙Ū)ᵀ(θ⊙Ū)  (can be precomputed once!)
        # Here we compute it on the fly once per evaluate(); 
        # for speed, you can cache R2 to opt and skip this every iter.
        temp = self.Uref_vec.copy()
        temp.pointwiseMult(temp, self.theta_vec)   # temp = θ⊙Ū
        # print(temp.array.max())
        R2 = temp.dot(temp)

        # (4) Ju = (J2/R2)
        Ju = J2 / R2

        # (5) derivative ∂Ju/∂U = (2/R2) * (r)
        rhs = r.copy()
        rhs.scale(2.0 / R2)
        # rhs.pointwiseMult(rhs, self.theta_vec)

        return Ju, rhs

    # ---------- 外部调用 ----------
    def evaluate(self):
        Ju, rhs = self._assemble_J_and_rhs()

        # solve adjoint  Kᵀ λ = rhs
        self.problem.set_adjoint_load(rhs)
        self.problem.solve_adjoint()

        # assemble  dJu/dρ  = -λᵀ ∂K/∂ρ U
        self.dJdrho_mat.zeroEntries()
        assemble_matrix(self.dJdrho_mat, self.dJdrho_form)
        self.dJdrho_mat.assemble()
        self.dJdrho_mat.mult(self.lam.vector, self.dJdrho_vec)

        # volume & grad wrt vf
        vol = self.comm.allreduce(
            assemble_scalar(self.V_form), op=MPI.SUM)
        V_ratio = vol / self.total_vol

        assemble_vector(self.dVdvf_vec, self.dVdvf_form)
        self.dVdvf_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                   mode=PETSc.ScatterMode.REVERSE)
        self.dVdvf_vec /= self.total_vol

        func_vals   = [Ju, V_ratio]
        sensitivities = [self.dJdrho_vec]     # dJu/dρ
        return func_vals, sensitivities, self.dVdvf_vec
