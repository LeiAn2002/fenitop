import numpy as np
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import (VectorFunctionSpace, FunctionSpace, Function, Constant,
                         dirichletbc, locate_dofs_topological)

from fenitop.NN import NeuralNetwork
import torch

model = NeuralNetwork(input_size=3, hidden1_size=256, hidden2_size=256, output_size=6)
model.load_state_dict(torch.load('/shared/fenitop_for_virtualgrowth/fenitop/trained_model.pth'))
model.eval()

model = model.double()

fre_hints = np.array([0.3333, 0.3333, 0.3333])
fre_hints_tensor = torch.tensor(fre_hints, dtype=torch.float64)
with torch.no_grad():
    prediction = model(fre_hints_tensor)
prediction_numpy = prediction.numpy()


fre_hints2 = np.array([0.3333, 0.3333, 0.3333])
fre_hints_tensor2 = torch.tensor(fre_hints2, dtype=torch.float64, requires_grad = True)
prediction2 = model(fre_hints_tensor2)
gradients = []
for j in range(6):
    grad = torch.autograd.grad(prediction2[j], fre_hints_tensor2, retain_graph=True)[0]
    gradients.append(grad)

fre_hints3 = np.array([0.33331, 0.3333, 0.3333])
fre_hints_tensor3 = torch.tensor(fre_hints3, dtype=torch.float64)
with torch.no_grad():
    prediction3 = model(fre_hints_tensor3)
prediction_numpy3 = prediction3.numpy()

fe = (prediction_numpy3[0] - prediction_numpy[0])/1e-5
# print(fe)
# print(gradients[0][0])
print((fe - gradients[0][0]) / gradients[0][0])