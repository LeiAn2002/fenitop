import ufl
from mpi4py import MPI
from dolfinx.fem import form, assemble_scalar, Function, FunctionSpace, petsc
from dolfinx.fem.petsc import create_vector, create_matrix, assemble_vector, assemble_matrix
from petsc4py import PETSc
from fenitop.NN import NeuralNetwork
import numpy as np

import torch

class Sensitivity_ksi():
    def __init__(self, opt, problem, u_field, lambda_field, ksi_phys_field_list, c_field_list, local_vf_phys_field):

        self.block_types = opt["block_types"]
        self.ksi_phys_field_list = ksi_phys_field_list
        self.local_vf_phys_field = local_vf_phys_field
        self.c_field_list = c_field_list
        self.dfdc_form_list = []
        self.dfdc_mat_list = []
        self.dUdc_vec_list = []
        self.dcdksi_mat_list = [[None for _ in range(6)] for _ in range(self.block_types+1)] # +1 is for volume fraction
        self.dUdksi_vector_list = [[None for _ in range(6)] for _ in range(self.block_types+1)]
        self.dCdksi_vec_list = [[None for _ in range(6)] for _ in range(self.block_types+1)]

        self.dCdc_form_list = []
        self.dCdc_vec_list = []
        self.opt_compliance = opt["opt_compliance"]

        num_ele = len(self.ksi_phys_field_list[0].x.array.copy())

        for i in range(6):
            self.dfdc_form_list.append(form(ufl.adjoint(ufl.derivative(opt["f_int"], c_field_list[i]))))
            self.dfdc_mat_list.append(create_matrix(self.dfdc_form_list[i]))
            self.dUdc_vec_list.append(c_field_list[i].vector.copy())

            self.dCdc_form_list.append(form(-ufl.derivative(opt["compliance"], c_field_list[i])))
            self.dCdc_vec_list.append(create_vector(self.dCdc_form_list[i]))

        for i in range(self.block_types+1):
            for j in range(6):
                self.dUdksi_vector_list[i][j] = c_field_list[i].vector.copy()
                self.dUdksi_vector_list[i][j].zeroEntries()
                self.dUdksi_vector_list[i][j].assemble()

                self.dCdksi_vec_list[i][j] = c_field_list[i].vector.copy()
                self.dCdksi_vec_list[i][j].zeroEntries()
                self.dCdksi_vec_list[i][j].assemble()

                petsc_mat = PETSc.Mat().create()
                petsc_mat.setSizes([num_ele, num_ele])
                petsc_mat.setType('aij')
                petsc_mat.setUp()
                self.dcdksi_mat_list[i][j] = petsc_mat

        self.problem, self.l_vec = problem, opt["l_vec"]
        self.u_field, self.lambda_field = u_field, lambda_field
        self.prod_vec = u_field.vector.copy()

        form(ufl.adjoint(ufl.derivative(opt["f_int"], c_field_list[i])))


    def evaluate(self):

        for i in range(6):
            with self.dCdc_vec_list[i].localForm() as loc:
                loc.set(0)
            assemble_vector(self.dCdc_vec_list[i], self.dCdc_form_list[i])
            self.dCdc_vec_list[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        model = NeuralNetwork(input_size=self.block_types+1, hidden1_size=256, hidden2_size=256, output_size=6)
        model.load_state_dict(torch.load('/shared/fenitop_for_virtualgrowth/fenitop/trained_model.pth'))
        model.eval()
        model = model.double()

        ksi_value_list = []
        for i in range(self.block_types):
            ksi_value_list.append(self.ksi_phys_field_list[i].x.array)
        ksi_value_list.append(self.local_vf_phys_field.x.array)

        for i in range(len(ksi_value_list[1])):
            fre_hints_and_vf = np.array([row[i] for row in ksi_value_list])
            fre_hints_and_vf_tensor = torch.tensor(fre_hints_and_vf, dtype=torch.float64, requires_grad = True)
            prediction = model(fre_hints_and_vf_tensor)
            gradients = []
            for j in range(6):
                grad = torch.autograd.grad(prediction[j], fre_hints_and_vf_tensor,
                                           retain_graph=True)[0]
                gradients.append(grad)
                for k in range(self.block_types+1):
                    self.dcdksi_mat_list[k][j].setValue(i, i, gradients[j][k])
                    self.dcdksi_mat_list[k][j].assemble()

        for i in range(self.block_types+1):
            for j in range(6):

                self.dCdc_vec_list[j].assemble()
                temp_vec2 = self.dCdc_vec_list[j].duplicate()
                temp_vec2.zeroEntries()
                self.dcdksi_mat_list[i][j].mult(self.dCdc_vec_list[j], temp_vec2)
                self.dCdksi_vec_list[i][j] = temp_vec2
                self.dCdksi_vec_list[i][j].assemble()
        
        if not self.opt_compliance:
            self.problem.solve_adjoint()

            for i in range(6):
                self.dfdc_mat_list[i].zeroEntries()
                assemble_matrix(self.dfdc_mat_list[i], self.dfdc_form_list[i])
                self.dfdc_mat_list[i].assemble()
                self.dfdc_mat_list[i].mult(self.lambda_field.vector, self.dUdc_vec_list[i])

            for i in range(self.block_types+1):
                for j in range(6):

                    self.dcdksi_mat_list[i][j].assemble()
                    self.dUdc_vec_list[j].assemble()
                    temp_vec = self.dUdc_vec_list[j].duplicate()
                    temp_vec.zeroEntries()
                    self.dcdksi_mat_list[i][j].mult(self.dUdc_vec_list[j], temp_vec)
                    self.dUdksi_vector_list[i][j] = temp_vec
                    self.dUdksi_vector_list[i][j].assemble()
        else:
            for i in range(self.block_types+1):
                for j in range(6):
                    self.dUdksi_vector_list[i][j] = None

        sensitivities_list = self.dUdksi_vector_list

        # return sensitivities_list
        dCdksi_list = self.dCdksi_vec_list

        return sensitivities_list, dCdksi_list


