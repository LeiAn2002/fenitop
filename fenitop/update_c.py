import numpy as np
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import (VectorFunctionSpace, FunctionSpace, Function, Constant,
                         dirichletbc, locate_dofs_topological)

from fenitop.NN import NeuralNetwork
import torch


class FieldUpdater:
    def __init__(self, opt, ksi_phys_field_list, c_field_list, local_vf_phys_field):
        self.local_vf_phys_field = local_vf_phys_field
        self.ksi_phys_field_list = ksi_phys_field_list
        self.c_field_list = c_field_list
        self.block_types = opt["block_types"]

    def update(self):
        ksi_value_list = []
        c_value_list = []

        for i in range(self.block_types):
            ksi_value_list.append(self.ksi_phys_field_list[i].x.array.copy())
        ksi_value_list.append(self.local_vf_phys_field.x.array.copy())

        for i in range(6):
            c_value_list.append(self.c_field_list[i].x.array.copy())

        path = '/shared/fenitop_for_virtualgrowth/fenitop/trained_model.pth'
        model = NeuralNetwork(input_size=self.block_types+1, hidden1_size=256,
                              hidden2_size=256, output_size=6)
        model.load_state_dict(torch.load(path))
        model.eval()
        model = model.double()

        for i in range(len(ksi_value_list[1])):
            fre_hints_and_vf = np.array([row[i] for row in ksi_value_list])
            fre_hints_and_vf_tensor = torch.tensor(fre_hints_and_vf, dtype=torch.float64)
            with torch.no_grad():
                prediction = model(fre_hints_and_vf_tensor)
            prediction_numpy = prediction.numpy()

            for j in range(6):
                c_value_list[j][i] = prediction_numpy[j]

        for i in range(6):
            self.c_field_list[i].x.array[:] = c_value_list[i].copy()
