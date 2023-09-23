import torch
from torch.autograd import Variable
import numpy as np
from typing import List
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class MyNeuralNet(torch.nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 num_layers: int = 10,
                 num_neurons: int = 50,
                 act_function: torch.nn.Module = torch.nn.Tanh()
                ) -> None:
        """
        My Deep Neural Network: A fully connected, customizable NeuralNetwork

        Args:
            num_inputs : int : the number of inputs (coordinates, variables) involved in the solution of the PDE
            num_outputs : int : the number of outputs (functions) solutions of the PDE
            num_layers (int, optional = 10): the number of hidden layers
            num_neurons (int, optional = 50): the number of neurons for each hidden layer
            act_function (torch.nn.Module, optional = torch.nn.Tanh): the non-linear activation function
            
        Returns:
            None
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.act_function = act_function

        layers = []
        # input layer
        # Many authors use or not an activation function in the input layer
        layers.extend([torch.nn.Linear(self.num_inputs, self.num_neurons), self.act_function])
        # hidden layers
        for _ in range(num_layers):
            layers.extend([torch.nn.Linear(self.num_neurons, self.num_neurons), self.act_function])
        # output layer
        # no activation needed in the output
        layers.append(torch.nn.Linear(self.num_neurons, self.num_outputs))

        # build the network
        self.network = torch.nn.Sequential(*layers)

    def forward(self, inputs : List[torch.Tensor]) -> torch.Tensor:
        """
        forward pass: 

        Args:
        inputs : List(torch.Tensor) : coordinates, variables involved in the solution of the PDE

        Returns:
        output : torch.Tensor : The output of the model
        """
        
        inputs = torch.cat(inputs, axis = 1)
        return self.network(inputs)#.squeeze()
        
def Euler_2D(x, y, NeuralNet, rho):

    solution = NeuralNet([x, y]) # num inputs = 2 (x, y), num_outputs = 3 (u, v, p)
    u = solution[:,0] # Flux velocity coord x
    v = solution[:,1] # Flux velocity coord y
    p = solution[:,2] # Pressure

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0] # partial u, x
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0] # partial u, y
    
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0] # partial v, x
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0] # partial v, y
    
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0] # partial p, x
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0] # partial p, x

    incompresibility = u_x + v_y # partial u, x + partial v, y = 0
    Euler_u = (u * u_x + v * u_y) + p_x/rho # Du/Dt = - (grad P)_x/rho
    Euler_v = (u * v_x + v * v_y) + p_y/rho # Dv/Dt = - (grad P)_y/rho
    
    return incompresibility, Euler_u, Euler_v
    
def Navier_Stokes_3D(t, x, y, z, NeuralNet, rho, mu, g):

    solution = NeuralNet([x, y, z]) # num inputs = 4 (t, x, y, z), num_outputs = 4 (u, v, w, p)
    u = solution[:,0] # Flux velocity coord x
    v = solution[:,1] # Flux velocity coord y
    w = solution[:,2] # Flux velocity coord y
    p = solution[:,3] # Pressure

    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0] # partial u, t
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0] # partial u, x
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0] # partial u, y
    u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0] # partial u, z
    
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0] # partial v, t
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0] # partial v, x
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0] # partial v, y
    v_z = torch.autograd.grad(v.sum(), z, create_graph=True)[0] # partial v, z

    w_t = torch.autograd.grad(w.sum(), t, create_graph=True)[0] # partial w, t
    w_x = torch.autograd.grad(w.sum(), x, create_graph=True)[0] # partial w, x
    w_y = torch.autograd.grad(w.sum(), y, create_graph=True)[0] # partial w, y
    w_z = torch.autograd.grad(w.sum(), z, create_graph=True)[0] # partial w, z
    
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0] # partial p, x
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0] # partial p, x
    p_z = torch.autograd.grad(p.sum(), z, create_graph=True)[0] # partial p, x

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0] # partial u, xx
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0] # partial u, yy
    u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0] # partial u, zz
    
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0] # partial v, xx
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0] # partial v, yy
    v_zz = torch.autograd.grad(v_z.sum(), z, create_graph=True)[0] # partial v, zz

    w_xx = torch.autograd.grad(w_x.sum(), x, create_graph=True)[0] # partial w, xx
    w_yy = torch.autograd.grad(w_y.sum(), y, create_graph=True)[0] # partial w, yy
    w_zz = torch.autograd.grad(w_z.sum(), z, create_graph=True)[0] # partial w, zz
    

    incompresibility = u_x + v_y + w_z # partial u, x + partial v, y + partial w, z= 0
    Navier_Stokes_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x/rho - mu * (u_xx + u_yy + u_zz)     # Du/Dt = - grad (P)_x/rho + mu * laplacian (u)
    Navier_Stokes_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y/rho - mu * (v_xx + v_yy + v_zz)     # Dv/Dt = - grad (P)_y/rho + mu * laplacian (v)
    Navier_Stokes_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z/rho - mu * (w_xx + w_yy + w_zz) - g # Dw/Dt = - grad (P)_z/rho + mu * laplacian (w) + g
    
    return incompresibility, Navier_Stokes_u, Navier_Stokes_v, Navier_Stokes_w
    
def train_Euler(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0
    mse = torch.nn.MSELoss()

    for i, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()

        outputs = model([x,y])

        # Loss based on PDE
        pde_output = Euler_2D(x, y, model, rho)
        mse_pde = mse(pde_output, torch.zeros(pde_output.shape))

        # Loss based on BoundaryConditions
        # Sampling points in boundary
        bound_points = sample(b_x, b_y) # Numpy arrays, return torch.tensor
        b_x = Variable(torch.from_numpy(b_x).float(), requires_grad=False).to(device)
        b_y = Variable(torch.from_numpy(b_y).float(), requires_grad=False).to(device)
        # Compute NN output in boundary
        bound_output = model([b_x, b_y])
        mse_bound = mse(bound_output, bound_points)

        # Combine loss functions
        
        loss = mse_pde + mse_bound
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 50 == 0:
            print(f"i: {i}, loss: {loss}")

    return epoch_loss
