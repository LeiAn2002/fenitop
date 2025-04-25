import ufl
from mpi4py import MPI
from dolfinx.fem import form, assemble_scalar, Function, FunctionSpace, petsc
from dolfinx.fem.petsc import create_vector, create_matrix, assemble_vector, assemble_matrix
from petsc4py import PETSc
from fenitop.NN import NeuralNetwork
import numpy as np
from fenitop.update_c import FieldUpdater
import torch


class Sensitivity_check_cksi():
    def __init__(self, opt, problem, u_field, lambda_field, ksi_phys_field_list, c_field_list):

        self.ksi_phys_field_list = ksi_phys_field_list
        self.c_field_list = c_field_list
        self.dfdc_form_list = []
        self.dfdc_mat_list = []
        self.dUdc_vec_list = []
        self.dcdksi_mat_list = [[None for _ in range(6)] for _ in range(3)]
        self.dUdksi_vector_list = [None for _ in range(3)]
        self.dCdksi_vec_list = [None for _ in range(3)]

        self.dCdc_form_list = []
        self.dCdc_vec_list = []

        num_ele = len(self.ksi_phys_field_list[0].x.array.copy())

        for i in range(6):
            self.dfdc_form_list.append(form(ufl.adjoint(ufl.derivative(opt["f_int"], c_field_list[i]))))
            self.dfdc_mat_list.append(create_matrix(self.dfdc_form_list[i]))
            self.dUdc_vec_list.append(c_field_list[i].vector.copy())

            self.dCdc_form_list.append(form(-ufl.derivative(opt["compliance"], c_field_list[i])))
            self.dCdc_vec_list.append(create_vector(self.dCdc_form_list[i]))

        for i in range(3):
            self.dUdksi_vector_list[i] = c_field_list[i].vector.copy()
            self.dUdksi_vector_list[i].zeroEntries()
            self.dUdksi_vector_list[i].assemble()

            self.dCdksi_vec_list[i] = c_field_list[i].vector.copy()
            self.dCdksi_vec_list[i].zeroEntries()
            self.dCdksi_vec_list[i].assemble()

            for j in range(6):
                petsc_mat = PETSc.Mat().create()
                petsc_mat.setSizes([num_ele, num_ele])
                petsc_mat.setType('aij')
                petsc_mat.setUp()
                self.dcdksi_mat_list[i][j] = petsc_mat

        self.problem, self.l_vec = problem, opt["l_vec"]
        self.u_field, self.lambda_field = u_field, lambda_field
        self.prod_vec = u_field.vector.copy()
        self.c_update = FieldUpdater(self.ksi_phys_field_list, self.c_field_list)

    def evaluate(self):

        self.problem.solve_adjoint()

        for i in range(6):
            self.dfdc_mat_list[i].zeroEntries()
            assemble_matrix(self.dfdc_mat_list[i], self.dfdc_form_list[i])
            self.dfdc_mat_list[i].assemble()
            self.dfdc_mat_list[i].mult(self.lambda_field.vector, self.dUdc_vec_list[i])

            assemble_vector(self.dCdc_vec_list[i], self.dCdc_form_list[i])
            self.dCdc_vec_list[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        model = NeuralNetwork(input_size=3, hidden1_size=256, hidden2_size=256, output_size=6)
        model.load_state_dict(torch.load('/shared/fenitop_for_virtualgrowth/fenitop/trained_model.pth'))
        model.eval()

        ksi_value_list = []
        for i in range(3):
            ksi_value_list.append(self.ksi_phys_field_list[i].x.array)

        for i in range(len(ksi_value_list[1])):
            fre_hints = np.array([ksi_value_list[0][i],ksi_value_list[1][i],ksi_value_list[2][i]])
            fre_hints_tensor = torch.tensor(fre_hints, dtype=torch.float32, requires_grad = True)
            prediction = model(fre_hints_tensor)
            gradients = []
            for j in range(6):
                grad = torch.autograd.grad(prediction[j], fre_hints_tensor, retain_graph=True)[0]
                gradients.append(grad)
                for k in range(3):
                    self.dcdksi_mat_list[k][j].setValue(i, i, gradients[j][k])
                    self.dcdksi_mat_list[k][j].assemble()

        c_value_old = self.c_field_list[0].x.array.copy()[0]
        self.ksi_phys_field_list[0].x.array[0] += 1e-4
        self.c_update.update()
        femethod = (self.c_field_list[0].x.array[0] - c_value_old)/1e-4
        
        error_value = femethod - self.dcdksi_mat_list[0][0].getValue(0, 0)
        # error_value = self.dcdksi_mat_list[0][0].getValue(0, 0)

        return error_value
