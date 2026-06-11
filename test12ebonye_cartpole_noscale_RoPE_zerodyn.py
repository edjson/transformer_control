### set cuda visible device to gpu 2 and 3
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import matplotlib.pyplot as plt
import workCon
# import workCon_linearsys as workCon
# import os
from eval import get_model_from_run
from tqdm import tqdm
import ipdb
import traceback
import random
import math
import generate_dataset
from scipy.integrate import solve_ivp
import pickle
import re



plot_label = 'mse_control'
phase_plot_label = 'mse_control_phaseplot'
mse_plot_label = 'mse_control_mseplot'
save_results = "trainsteps_test_mse_control.txt"
save_phase_plot = "trainsteps_test_mse_control.txt"
log_info = "trainsteps_log_mse_control.txt"
model_name= "cartpole_cos_sin_theta" #"cartpole_cos_sin_theta" #"cartpole" #"test"
model_run_id= "fc5cb0fe-d4c6-4fba-8196-01a37d611194" #"5a28bfa8-c81f-473b-996f-879dfadc62df" #"3f8379cf-1cf0-47db-9210-5abb1c43e5fb" #"b3725997-9aee-4578-b668-d33e7cb29c4e" #"2ca9672c-582e-43ef-85cf-8550f325947a" #"a1d5f223-6768-4134-934b-4879031f7ea1" #"cf756e46-3ddb-4df7-9a13-ba850f495257" #"5b73d6c9-b526-4bfe-bab3-005f5369cf5a" #"a1d5f223-6768-4134-934b-4879031f7ea1" #"c7978c77-a2d6-4d87-a256-d42690204829" #"f8211c69-7ae8-47b3-9bd2-e4f0edda1e82" #"de00c432-078f-44d1-8cc9-e6ab0dfdca88" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"c3af70ca-3733-4cec-a876-95db9bc9a593" #"32ea0675-5539-4d02-80fb-7bfe1f4c263e" #"57f687d9-9e41-48f5-83d4-559592ca762b" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"be268b82-d25e-4026-b303-91ba3c6d1e9f" 
# model_checkpoint_step= 40000 #30000 #55000 #274941 #195000 #5000 #15000 #274941 #115000 #300800 #204800 #102400
model_checkpoint_epoch = 1 #25 #59 #125 #14 #38 #60 #125
# folder_name = f"inference_run/{plot_label}_{model_checkpoint_step}_{model_run_id}"
mode = 'ood' # 'train', 'ood', 'indistr'

    

total_time = 14 #4 #5 #1.5
dt = 0.025
# Num_of_context = 30 #50 #150
Num_of_pendulums = 100 #5 #100 #200 #1 #10 #20 #40 #10
# start_index_num = [Num_of_context]




random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1000)

# def mse(theta_model, thetadot_model, theta_rk4, thetadot_rk4, device):
def mse(xs_pred, xs_true, device):
    """
    Calculates the Mean Squared Error (MSE) between the predicted and true states.

    Args:
        theta_model (np.ndarray or list): Model's predicted theta 
        thetadot_model (np.ndarray or list): Model's predicted thetadot 
        theta_rk4 (np.ndarray or list): True theta using RK4
        thetadot_rk4 (np.ndarray or list): True thetadot using RK4

    Returns:
        float: The computed MSE value, representing the average squared difference between 
        the predicted and true states.
    """
    # xs_pred = torch.tensor(np.column_stack((theta_model, thetadot_model)), dtype=torch.float32, device=device)
    # xs_true = torch.tensor(np.column_stack((theta_rk4, thetadot_rk4)), dtype=torch.float32, device=device)
    # xs_pred = torch.stack((theta_model, thetadot_model), dim=1).to(device)
    # xs_true = torch.stack((theta_rk4, thetadot_rk4), dim=1).to(device)
    xs_pred = xs_pred.to(device)
    xs_true = xs_true.to(device)
    return (xs_true - xs_pred).pow(2).mean().item()

def mse_controls(control_values_model, control_values_rk4, device):
    """
    Calculates the Mean Squared Error (MSE) between the predicted and true control values.

    Args:
        control_values_model (np.ndarray or list): Model's predicted control values
        control_values_rk4 (np.ndarray or list): True control values using RK4

    Returns:
        float: The computed MSE value, representing the average squared difference between 
        the predicted and true control values.
    """
    # control_pred = torch.tensor(control_values_model, dtype=torch.float32, device=device)
    # control_true = torch.tensor(control_values_rk4, dtype=torch.float32, device=device)
    control_pred = control_values_model.to(device)
    control_true = control_values_rk4.to(device)
    # return (control_true - control_pred).pow(2).mean().item()
    #mean absolute error instead
    return torch.abs(control_true - control_pred).mean().item()

# def load_model(run_dir, name, run_id, step):
def load_model(run_dir, name, run_id, step, epoch):
    """
    Loads a pre-trained model and its configuration from a specified run directory.

    Args:
        run_dir (str): The base directory containing the model runs.
        name (str): The name of the model
        run_id (str): The unique identifier for the specific run to load the model from.
        step (int, optional): The training step at which to load the model checkpoint. 

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The loaded model.
            - conf (dict): The configuration dictionary associated with the model run.
    """
    run_path = os.path.join(run_dir, name, run_id)
    model, conf = get_model_from_run(run_path, epoch= epoch, step=step)
    return model, conf


def debug_model(model, XData, YS, device):
    """
    Debugs the model by running a forward pass with the provided input data.

    Args:
        model (torch.nn.Module): The model to be debugged.
        XData (torch.Tensor): The input data for the model.
        YS (torch.Tensor): The control input data.
        device (torch.device): The device on which the model is located.

    Returns:
        tuple: A tuple containing:
            - u_pred (torch.Tensor): The predicted control input.
            - state_pred (torch.Tensor): The predicted state.
    """
    
    # XData = [XData[i]/(400*0.98**i) for i in range(len(XData))]
    # XData = torch.stack(XData, axis=0)
    # XData = arcsinh_scaling(XData)
    # YS = [YS[i]/(400*0.98**i) for i in range(len(YS))]
    # YS = torch.stack(YS, axis=0)
    # YS = arcsinh_scaling(YS)
    model.eval()
    XData = XData.to(device)
    YS = YS.to(device)
    with torch.no_grad():
        model = model.to(device)
        u_pred, state_pred = model(XData, YS, inf="yes")
    # import pdb; pdb.set_trace()
    return u_pred, state_pred



# def make_deriv_vectorized_simple(m_c, m_1, l_1, g=9.81):
#     """
#     Create a batched derivative function without control logic.
#     Inputs:
#         m_c, m_1, l_1: (batch,) tensors
#     Returns:
#         deriv(state, u): computes ẋ = f(x, u)
#     """
#     def deriv(state, u):
#         x, theta1, x_dot, theta1_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
#         u = u.squeeze(0)
#         cos_theta1 = torch.cos(theta1)
#         sin_theta1 = torch.sin(theta1)

#         # import pdb; pdb.set_trace()
        

#         A11 = m_c.unsqueeze(-1) + m_1.unsqueeze(-1) 
#         A12 = m_1.unsqueeze(-1) * l_1.unsqueeze(-1) * cos_theta1
#         A21 = A12
#         A22 = m_1.unsqueeze(-1) * l_1.unsqueeze(-1) ** 2

#         # import pdb; pdb.set_trace()

#         A = torch.stack([
#             torch.stack([A11, A12], dim=1),
#             torch.stack([A21, A22], dim=1)
#         ], dim=1)  # (batch, 2, 2)
#         # import pdb; pdb.set_trace()

#         b1 = u + m_1 * l_1 * sin_theta1 * theta1_dot**2
#         b2 = -m_1 * l_1 * g * sin_theta1
#         # b2 = b2.unsqueeze(-1)  # (batch, 1)

#         # import pdb; pdb.set_trace()
#         b = torch.stack([b1, b2], dim=1)  # (batch, 2)
#         b = b.to(torch.float64)

#         # print(f"A shape: {A.shape}")
#         # print(f"b shape: {b.shape}")

#         # import pdb; pdb.set_trace()

#         accel = torch.linalg.solve(A, b).squeeze(-1)  # (batch, 2)

#         dydt = torch.zeros_like(state)
#         dydt[:, 0] = x_dot
#         dydt[:, 1] = theta1_dot
#         dydt[:, 2] = accel[:, 0]
#         dydt[:, 3] = accel[:, 1]

#         return dydt

#     return deriv

def cartpole_dynamics(state, u, cartmass, polemass, polelength, g=9.81):
    """
    Computes the dynamics of the cartpole system.
    Inputs:
        state: (4,) tensor representing [x, x_dot, theta, theta_dot]
        u: (1,) tensor representing the control input (force applied to the cart)
        cartmass: float, mass of the cart
        polemass: float, mass of the pole
        polelength: float, length of the pole
        g: float, acceleration due to gravity (default 9.81 m/s^2)
    Returns:
        dydt: (4,) tensor representing the derivatives [x_dot, xacc, theta_dot, thetaacc]
    """
    x, x_dot, theta, theta_dot = state
    force = u.item()  # Convert tensor to float

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = (force + sintheta * polelength * polemass * theta_dot**2) / (cartmass + polemass)
    thetaacc = (g * sintheta - costheta * temp) / (polelength * (4.0 / 3.0 - polemass * costheta**2 / (cartmass + polemass)))
    xacc = temp - (polemass * polelength * thetaacc * costheta) / (cartmass + polemass)

    return torch.tensor([x_dot, xacc, theta_dot, thetaacc], dtype=torch.float32).to(state.device)

# def rk4_step(state, u, dt, dynamics):
def rk4_step(state, u, dt, dynamics, cartmass, polemass, polelength):
    """
    Perform a single RK4 step.
    Inputs:
        state: (batch, 4)
        u: (batch,)
        dt: float
        dynamics: function that takes (state, u) and returns derivative
    Returns:
        state_next: (batch, 4)
    """
    k1 = dynamics(state, u, cartmass, polemass, polelength)
    k2 = dynamics(state + 0.5 * dt * k1, u, cartmass, polemass, polelength)
    k3 = dynamics(state + 0.5 * dt * k2, u, cartmass, polemass, polelength)
    k4 = dynamics(state + dt * k3, u, cartmass, polemass, polelength)
    state_next = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # state_next = state_next.squeeze()
    return state_next





def run_inference_on_model(model, XData, YS, total_time, device, dt=0.01, context=1, start_index=1, cartmass=1.0, polemass=0.1, polelength=0.5):
    """
    Runs inference on a trained model to simulate the dynamics of a pendulum system over time, given an initial state and context data.

    Args:
        model (torch.nn.Module): The trained model used for predicting control inputs.
        XData (torch.Tensor): The input state  (e.g., [theta, thetadot])
        YS (torch.Tensor): The ground truth control input data
        total_time (float): Total duration of the simulation in seconds.
        dt (float, optional): Time step for the simulation. Defaults to 0.01.
        context (int, optional): The number of previous time steps used as context for the model. Defaults to 1.
        start_index (int, optional): The starting index for inference. Must be at least equal to `context`. Defaults to 1.
        device (torch.device): The device on which the model and data are located (e.g., 'cpu' or 'cuda').
        cartmass (float, optional): Mass of the cart. Defaults to 1.0.
        polemass (float, optional): Mass of the pendulum. Defaults to 0.1.
        polelength (float, optional): Length of the pendulum. Defaults to 0.5.
    Returns:
        tuple: A tuple containing:
            - T (np.ndarray): Array of time steps during the simulation.
            - theta_model (np.ndarray): Array of predicted theta over time.
            - thetadot_model (np.ndarray): Array of predicted thetadot over time.
            - YS_context (np.ndarray): Array of control inputs over time.

    Raises:
        AssertionError: If `start_index` is less than `context`.
    """
    # start_index = start_index_num[0]
    assert start_index >= context, "start_index must be at least equal to context"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    cartmass = torch.tensor(cartmass).to(device) #, dtype=torch.float32, device=device)
    polemass = torch.tensor(polemass).to(device) #, dtype=torch.float32, device=device)
    polelength = torch.tensor(polelength).to(device) #, dtype=torch.float32, device=device)
    
    # import pdb; pdb.set_trace()
    T = np.arange(0, total_time, dt)
    n_steps = len(T)

    
   
    # XData_context = XData[:start_index].to(device)
    XData_context = XData[:context].to(device) #6/25/2025
    


    
   
    
    # YS_context = YS[:start_index].to(device)
    YS_context = YS[:context-1].to(device) if context > 1 else torch.empty((0,2), device = device)  #6/25/2025

    counter = 0 if context == 1 else 1 #6/25/2025
    switch = 0

    XData_final = XData_context.clone() #6/25/2025
    YS_final = YS_context.clone() #6/25/2025

    states_scale = [7.0, 8.0, 1.0, 1.0, 5.0] # x, x_dot, cos(theta), sin(theta), theta_dot 2/22/2026
    control_scale = 15.0 # 2/22/2026

    

    for i in range(start_index, n_steps):
        with torch.no_grad():
            # u_pred = model(XData_context[start_index - context:], YS_context[start_index - context:], inf = "yes")
            # XData_context_scaled = XData_context / torch.tensor(states_scale, device=device) #6/25/2025
            # YS_context_scaled = YS_context / control_scale #6/25/2025
            xs = XData_context[start_index - context:]
            cos_theta = torch.cos(xs[:, 2]) #7/26/2025
            sin_theta = torch.sin(xs[:, 2]) #7/26/2025
            xs = torch.cat((xs[:, :2], cos_theta.unsqueeze(1), sin_theta.unsqueeze(1), xs[:, 3:]), dim=-1) #7/26/2025
            xs = xs / torch.tensor(states_scale, device=device) #7/26/2025
            ys = YS_context[start_index - context:]
            ys = ys / torch.tensor([control_scale, 1.0], device=device)  # scale control and keep switch label the same 7/26/2025
            ys_for_model = ys.clone()
            # add 1 to switch labels so that 0 becomes 1, 1 becomes 2, and 2 becomes 3 (0: zero dynamics, 1: swing up, 2: lqr)
            ys_for_model[:, 1] = ys_for_model[:, 1] + 1.0 
            # import pdb; pdb.set_trace()
            # print(f"xs shape: {xs.shape}, ys shape: {ys.shape}")
            # u_pred, state_pred = model(xs, ys, inf = "yes") # 7/18/2025
            # debugging
            # print(f"ys_for_model shape: {ys_for_model.shape}") #7/18/2025
            # print(f"control in ys_for_model: {ys_for_model[:, 0]}") #7/18/2025
            # print(f"switch labels in ys_for_model: {ys_for_model[:, 1]}") #7/18/2025
            u_pred, state_pred, flag_pred = model(xs, ys_for_model, inf = "yes") # 7/18/2025
            # u_pred, state_pred = model(XData_context[start_index - context:], YS_context[start_index - context:], inf = "yes")
            # u_pred = model(XData_context_scaled, YS_context_scaled, inf = "yes")
            # u = u_pred[0][-2].cpu().numpy()
            # print(f"u_pred shape: {u_pred.shape}, flag_pred shape: {flag_pred.shape}")
            u = u_pred[0][-1] #.cpu().numpy()
            # u = u_pred[0][-1][0]
            # u_with_label = u_pred[0][-1]
            flag_pred = flag_pred[0][-1] # 7/18/2025
            flag_pred = torch.argmax(flag_pred)-1 #.float()
            if i >= context:
                # If the model is stuck in 'identification' mode (-1) 
                # but we are past the context window, force it to 'swing-up' (0)
                if flag_pred == -1:
                    actual_mode_to_use = torch.tensor(0, device=device)
                else:
                    actual_mode_to_use = flag_pred
            else:
                actual_mode_to_use = torch.tensor(-1, device=device)
             # 7/18/2025
            # flag_prob = torch.sigmoid(flag_pred) # 7/24/2025
            # flag_pred = 1 if flag_pred > 0.5 else 0
            # flag_pred = (flag_prob > 0.5).int() #.float() # 7/24/2025

            # if switch == 1:
            #     flag_pred = torch.tensor([1.0], device=device)

            # elif flag_pred == 1:
            #     switch = 1
            
            

            
            # import pdb; pdb.set_trace()
            # print(f"u: {u}, flag_pred: {flag_pred}") #7/18/2025
            # u_with_label = torch.cat((u.unsqueeze(0), flag_pred.unsqueeze(0)), dim=0) # 7/18/2025
            # u_with_label = u_with_label.squeeze(-1)
            # u_with_label = torch.cat((u * control_scale, flag_pred.unsqueeze(0)), dim=0) # 7/18/2025
            u_with_label = torch.cat((u * control_scale, actual_mode_to_use.unsqueeze(0)), dim=0) 
            u_with_label = u_with_label.squeeze(-1) #6/25/2025
            
        # import pdb; pdb.set_trace()
        # print(f"u with label shape: {u_with_label.shape}")
    
        
        
        # u_unscaled = undo_new_scaling_controls(u, index=i-1).cpu().numpy()
        u_unscaled = u * control_scale #.cpu().numpy()
        previous_state_unscaled = XData_context[-1] #.cpu().numpy()

        # next_state = rk4_step(
        #     previous_state_unscaled.unsqueeze(0),
        #     u_unscaled.unsqueeze(0),
        #     dt,
        #     make_deriv_vectorized_simple(cartmass, polemass, polelength)
        # )

        next_state = rk4_step(
            previous_state_unscaled,
            u_unscaled,
            dt,
            cartpole_dynamics,
            cartmass=cartmass,
            polemass=polemass,
            polelength=polelength
        )

        new_X = next_state#.cpu().numpy()
        new_X = new_X.unsqueeze(0)

        # import pdb; pdb.set_trace()

        XData_context = torch.cat((XData_context, new_X), dim=0) ###### 2/11/2025 (ebonye): added [1:] to fix the context length (sliding window)
        XData_final = torch.cat((XData_final, new_X), dim=0) #6/25/2025
        # sliding window
        if XData_context.shape[0] > context:
            XData_context = XData_context[1:] # drop the oldest state to maintain the context length

        # print(f"new_X shape: {new_X.shape}, XData_context shape: {XData_context.shape}, XData_final shape: {XData_final.shape}") #6/25/2025
        new_Y = torch.tensor(u_with_label, dtype=torch.float32, device=device)
        # import pdb; pdb.set_trace()
        # new_Y = torch.cat([u, flag_tensor], dim=0) #6/24/2025

        
        
        if counter == 0 and context > 1: #6/25/2025
            # import pdb; pdb.set_trace()
            # YS_context = torch.cat((YS_context[:-1], new_Y), dim=0) ###### 2/11/2025 (ebonye): added [1:-1] to fix the context length
            # YS_context = torch.cat((YS_context[:-1], new_Y.unsqueeze(0)), dim=0) #6/24/2025
            YS_context = new_Y.unsqueeze(0) #6/24/2025
            YS_final = new_Y.unsqueeze(0) #6/25/2025
            # print(f"YS_context shape: {YS_context.shape}")
            counter = 1

            # print(YS_context.shape)
        elif counter == 0 and context == 1: #6/25/2025
            YS_context = torch.empty((0,2), device=device) #6/25/2025
            YS_final = new_Y.unsqueeze(0) #6/25/2025
            counter = 1
        else:
            # import pdb; pdb.set_trace()
            # YS_context = torch.cat((YS_context, new_Y), dim=0) ###### 2/11/2025 (ebonye): added [1:] to fix the context length

            YS_context = torch.cat((YS_context, new_Y.unsqueeze(0)), dim=0) #6/24/2025
            YS_final = torch.cat((YS_final, new_Y.unsqueeze(0)), dim=0) #6/25/2025
            if YS_context.shape[0] > context-1:
                YS_context = YS_context[1:] # drop the oldest control to maintain the context length
                

            # print(f"new_Y shape: {new_Y.shape}, YS_context shape: {YS_context.shape}, YS_final shape: {YS_final.shape}") #6/25/2025            

        # print(f"YS_context shape: {YS_context.shape}")
        # print(f"XData_context shape: {XData_context.shape}")


    # YS_context = YS_context #.cpu().numpy()
   
    # x_model = XData_context[:, 0] #.cpu().numpy()
    # theta_model = XData_context[:, 2] #.cpu().numpy()
    # xdot_model = XData_context[:, 1] #.cpu().numpy()
    # thetadot_model = XData_context[:, 3] #.cpu().numpy()

    YS_context = YS_final #.cpu().numpy()
    x_model = XData_final[:, 0] #.cpu().numpy()
    theta_model = XData_final[:, 2] #.cpu().numpy()
    xdot_model = XData_final[:, 1] #.cpu().numpy()
    thetadot_model = XData_final[:, 3] #.cpu().numpy()
    # import pdb; pdb.set_trace()
    
    
    return T, x_model, theta_model, xdot_model, thetadot_model, YS_context




def plot_mse_vs_context_length(mean_mse, std_mse, save_results_path, folder_name, plot_label, loss_type= "state"):
    """
    Plots the mean MSE vs. context length graph.

    Args:
        mean_mse (dict): A dictionary containing the mean MSE values for each context length.
        std_mse (dict): A dictionary containing the standard deviation of MSE values for each context length.
        save_results_path (str): The path to save the results log.
        folder_name (str): The directory where the plot will be saved.
        plot_label (str): The label for the plot.
        loss_type (str): The type of loss used for the MSE calculation. Defaults to "state mse".

    Returns:
        None
    """
    # with open(save_results_path, "a") as f:
    #     f.write(f"mean_mse: {mean_mse}\n")
    #     f.write(f"std_mse: {std_mse}\n")

    x_axis = list(mean_mse.keys())
    y_axis = list(mean_mse.values())
    y_err = list(std_mse.values())

    plt.figure(figsize=(10, 6))
    plt.errorbar(x_axis, y_axis, yerr=y_err, fmt='o-', capsize=8, label=plot_label)
    plt.xticks(x_axis)
    plt.xlabel('Context Length')
    plt.ylabel('Mean MSE')
    plt.title(f'Mean MSE vs Context Length {Num_of_pendulums} Pendulums')
    plt.legend()
    plt.grid(True)


    plot_path = os.path.join(folder_name, f"{loss_type}_mean_mse_vs_context_length({Num_of_pendulums} Pendulums).png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')





import os
import numpy as np
import matplotlib.pyplot as plt

def phase_plot(theta_plot, thetadot_plot, theta_rk4_unscaled, save_results_path, folder_name):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    # Create save directory if it doesn't exist
    save_dir = os.path.join(save_results_path, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))

    # Plot ground truth
    plt.plot(theta_rk4_unscaled[:-1], np.diff(theta_rk4_unscaled)/0.01, 
             label='Ground Truth', linewidth=3, color='black', alpha=0.8)

    # Plot every 10th sequence from theta_plot and thetadot_plot
    for idx in range(0, len(theta_plot), 10):
        plt.plot(theta_plot[idx], thetadot_plot[idx], linestyle='--', alpha=0.7, label=f'Context {idx+1}')

    # Labels and title
    plt.xlabel("Theta (rad)")
    plt.ylabel("Theta dot (rad/s)")
    plt.title("Phase Plot: Theta vs Theta dot")
    plt.legend(fontsize='small', ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Generate random number to ensure unique filename
    random_index = random.randint(1000, 9999)
    plot_path = os.path.join(save_dir, f"phase_plot_{random_index}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return plot_path

# Each plot will now be saved with a randomly generated unique identifier.


# Now each plot will be saved uniquely by specifying plot_index.


# Ready to plot when you provide your data.


def get_mass_length_Ks_from_text_file(file_path, target_iteration):
    """
    Reads the mass and length values from a text file.

    Args:
        file_path (str): The path to the text file containing the mass and length values.

    Returns:
        tuple: A tuple containing:
            - mass (float): The mass of the pendulum.
            - length (float): The length of the pendulum.
    """
    iteration_found = False
    iteration_data = {}
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            # check for iteration
            iteration_match = re.search(r'Iteration:\s*(\d+)', line)
            if iteration_match:
                current_iteration = int(iteration_match.group(1))
                if current_iteration == target_iteration:
                    iteration_found = True
                else:
                    iteration_found = False
            
            # once target iteration is found, extract mass, length, and K values
            if iteration_found:

                # extract mass and length
                mass_match = re.search(r'Masses:\s*(\d+\.\d+)', line)
                length_match = re.search(r'Lengths:\s*(\d+\.\d+)', line)
                K_match = re.search(r'K:\s*\[\[(.*?)\]\]', line)

                if mass_match:
                    iteration_data['mass'] = float(mass_match.group(1))
                if length_match:
                    iteration_data['length'] = float(length_match.group(1))

                if K_match:
                    # iteration_data['K'] = float(K_match.group(1))
                    K_values = list(map(float, K_match.group(1).split()))
                    iteration_data['K'] = torch.tensor(K_values).reshape(1, 2)  # Assuming K is 1x2 matrix
                
    return iteration_data['mass'], iteration_data['length'], iteration_data['K']



def main(
        chkpt_step, folder_name,
        number_of_context,
        phase_data,
        controls_data,
        data_and_controls,
        ):
    """_summary_
    """
    model, _ = load_model(
        run_dir="./models",
        name= model_name,
        run_id= model_run_id,
        # step=model_checkpoint_step,
        step=chkpt_step,
        epoch=model_checkpoint_epoch ###### 2/11/2025 (ebonye): added epoch
    )
    # folder_name = f"inference_run/{plot_label}_{chkpt_step}_{model_run_id}"




    os.makedirs(folder_name, exist_ok=True)
    save_results_path = os.path.join(folder_name, save_results)
    save_phase_path = os.path.join(folder_name, save_phase_plot)
    log_info_path = os.path.join(folder_name, log_info)
    # context_lengths = [0] * Num_of_context
    # start_indices = [Num_of_context]
    # contexts = np.arange(1, Num_of_context + 1)
    # contexts = [1, 10, 20, 30, 40, 50]
    # contexts = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    # contexts = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    # contexts = [1, 5, 10, 30, 60, 120]

    

    ####################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # pends = np.random.randint(0, 1000, size=Num_of_pendulums)
    # pends = [1050] 
    # masses = []
    # lengths = []
    # X0s = []
    # data_and_controls = []
    
    if mode == 'ood':
        # base_dir = f"/data/esmith/Dataset_LinearSystem_ICL/picklefolder_test_outofdistr"
        # base_dir = f"/data/esmith/Dataset_Cartpole_ICL/picklefolder_test_outofdistr"
        # base_dir = f"/data/esmith/Dataset_Cartpole_AIGymWithNoise_ICL/picklefolder_test_outofdistr"
        # base_dir = f"/data/esmith/Dataset_Cartpole_AIGymWithNoise_ICL_new_ranges/picklefolder_test_outofdistr"
        base_dir = f"/data/esmith/Dataset_Cartpole_AIGymWithNoise_ICL_new_ranges_zero_dyn/picklefolder_test_outofdistr"
        # pickle_file = "batch_test_0.pkl"
        pickle_file = f"batch_test_0_{number_of_context}.pkl"
    elif mode == 'indistr':
        # base_dir = f"/data/esmith/Dataset_LinearSystem_ICL/picklefolder_test_indistr"
        # base_dir = f"/data/esmith/Dataset_ConstantLinearSystem_ICL/picklefolder_test_indistr"
        # base_dir = f"/data/esmith/Dataset_Cartpole_ICL/picklefolder_test_indistr"
        # base_dir = f"/data/esmith/Dataset_Cartpole_Better_ICL/picklefolder_test_indistr"
        # base_dir = f"/data/esmith/Dataset_Cartpole_AIGym_ICL/picklefolder_test_indistr"
        # base_dir = f"/data/esmith/Dataset_Cartpole_AIGymWithNoise_ICL/picklefolder_test_indistr"
        # base_dir = f"/data/esmith/Dataset_Cartpole_AIGymWithNoise_ICL_new_ranges/picklefolder_test_indistr"
        base_dir = f"/data/esmith/Dataset_Cartpole_AIGymWithNoise_ICL_new_ranges_zero_dyn/picklefolder_test_indistr"
        # pickle_file = "batch_test_0.pkl"
        # pickle_file = "batch_test_1.pkl"
        pickle_file = f"batch_test_0_{number_of_context}.pkl"
    elif mode == 'train':
        # base_dir = f"/data/esmith/Dataset_LinearSystem_ICL/picklefolder"
        # base_dir = f"/data/esmith/Dataset_ConstantLinearSystem_ICL/picklefolder"
        # base_dir = f"/data/esmith/Dataset_Cartpole_ICL/picklefolder"
        # base_dir = f"/data/esmith/Dataset_Cartpole_Better_ICL/picklefolder"
        # base_dir = f"/data/esmith/Dataset_Cartpole_AIGym_ICL/picklefolder"
        # base_dir = f"/data/esmith/Dataset_Cartpole_AIGymWithNoise_ICL/picklefolder"
        base_dir = f"/data/esmith/Dataset_Cartpole_AIGymWithNoise_ICL_new_ranges/picklefolder"
        pickle_file = "batch_0.pkl"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    file_path_test_data = os.path.join(base_dir, pickle_file)
    with open(file_path_test_data, "rb") as f:
        xs, ys, cartmasses, polemasses, polelenghs = pickle.load(f)
        
    # goal_state = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], device=device)  # [x, x_dot, cos(theta), sin(theta), theta_dot]
    # goal_threshold = 0.1  # Threshold for goal state

    xs_cos_tensor = torch.cos(xs[:, :, 2])  # Cosine of theta
    xs_sin_tensor = torch.sin(xs[:, :, 2])  # Sine of theta
    xs_updated = torch.cat((xs[:, :, :2], xs_cos_tensor.unsqueeze(-1), xs_sin_tensor.unsqueeze(-1), xs[:, :, 3:]), dim=-1)  # Update xs with cos and sin of theta
    # mask = torch.norm(xs_updated[:, -1, :5] - goal_state, dim=-1) < goal_threshold  # Check if the last state is close to the goal state
    
    final_window = xs_updated[:, -40:, :]  # Get the last 40 time steps
    cos_theta_thresh = 0.9
    sin_theta_thresh = 0.5
    theta_dot_thresh = 1.0
    is_upright = (torch.abs(final_window[:, :, 2]) > cos_theta_thresh) & \
                (torch.abs(final_window[:, :, 3]) < sin_theta_thresh) & \
                (torch.abs(final_window[:, :, 4]) < theta_dot_thresh)  # Check if the pendulum is upright
    mask = is_upright.all(dim=1)  # Check if all time steps


    all_pends = torch.where(mask)[0].cpu().numpy()  # Get indices of pendulums that meet the goal state condition
    # choose pendulums that do not meet the goal state condition for more challenging inference
    # all_pends = torch.where(~mask)[0].cpu().numpy()  # Get indices of pendulums that do not meet the goal state condition
    
    # pends = np.random.choice(all_pends, size=Num_of_pendulums, replace=False)  # Randomly select pendulums from those that meet the goal state condition
    pends = np.arange(Num_of_pendulums)
    # import pdb; pdb.set_trace()

    cartmasses = [cartmasses[i] for i in pends]
    polemasses = [polemasses[i] for i in pends]
    polelenghs = [polelenghs[i] for i in pends]
    states = [xs[i].cpu().detach().numpy() for i in pends]
    # controls.append([ys[i] for i in pends])
    controls = [ys[i].cpu().detach().numpy() for i in pends]
    data_and_controls.append([(xs[i].cpu().numpy(), ys[i].cpu().numpy()) for i in pends])
    # for i in range(len(states)):
    #     print(f"final states {i}: {states[i][-1]}")

    # import pdb; pdb.set_trace()

       


   
    
    with open(log_info_path, "w") as file:
        # mse_results = {context_length: [] for context_length in contexts}
        # mse_control_results = {context_length: [] for context_length in contexts}
        # phase_data = {context_length: [] for context_length in contexts}
        # controls_data = {context_length: [] for context_length in contexts}
        counter = 0
        
        for cartmass, polemass, polelength in tqdm(zip(cartmasses, polemasses, polelenghs), desc="Cartpole", total=len(cartmasses), leave=False):
            # X0 = generate_random_X0()
            # X0 = [ 1.0852e+00, -1.3760e+00]
            # X0 = [1.4575, 0.1397]
            mse_per_context = []
            mse_control_per_context = []
            
            x_rk4 = states[counter][:, 0]
            theta_rk4 = states[counter][:, 2]
            xdot_rk4 = states[counter][:, 1]
            thetadot_rk4 = states[counter][:, 3]
            control_values_rk4 = controls[counter]

            
            

            

            # xs_dataset = np.column_stack((x1_rk4, x2_rk4))
            # xs_dataset = np.column_stack((x_rk4, theta_rk4, xdot_rk4, thetadot_rk4))
            xs_dataset = np.column_stack((x_rk4, xdot_rk4, theta_rk4, thetadot_rk4))
            xs_dataset = torch.tensor(xs_dataset).float().cuda()
            control_values_rk4 = torch.tensor(control_values_rk4).float().cuda()
            

            x_plot = []
            theta_plot = []
            xdot_plot = []
            thetadot_plot = []

           

            control_values_scaled = control_values_rk4
            states_scaled = xs_dataset
            
            ######

            # import pdb; pdb.set_trace()
            # for context in tqdm(contexts, desc="Context Loop", leave=False):
            # file.write(f"  Start Index: {start_index}\n")
            # theta_rk4_temp = theta_rk4[start_index-1:]
            # thetadot_rk4_temp = thetadot_rk4[start_index-1:]
            start_index = number_of_context

            # x1_rk4_temp = x1_rk4[context:]
            # x2_rk4_temp = x2_rk4[context:]
            x_rk4_temp = x_rk4[number_of_context:]
            theta_rk4_temp = theta_rk4[number_of_context:]
            xdot_rk4_temp = xdot_rk4[number_of_context:]
            thetadot_rk4_temp = thetadot_rk4[number_of_context:]
            controls_rk4_temp = control_values_rk4[number_of_context:]

            
            T_model, x_model2, theta_model2, xdot_model2, thetadot_model2, controls_model2 = run_inference_on_model(
                model, xs_dataset, control_values_rk4, total_time, device, dt,
                context=number_of_context, start_index=start_index, cartmass=cartmass, polemass=polemass, polelength=polelength
            )   


            
            # trajectory = torch.stack([x_model2, theta_model2, xdot_model2, thetadot_model2], axis=1)
            trajectory = torch.stack([x_model2, xdot_model2, theta_model2, thetadot_model2], axis=1)  # 7/26/2025
            controls_for_trajectory = controls_model2
            phase_data[number_of_context].append(trajectory)
            # import pdb; pdb.set_trace()
            controls_data[number_of_context].append(controls_for_trajectory)
            

            

            x_plot.append(x_model2)
            theta_plot.append(theta_model2)
            xdot_plot.append(xdot_model2)
            thetadot_plot.append(thetadot_model2)
            
            # state_pred_context = torch.stack([x_model2[context:], theta_model2[context:], xdot_model2[context:], thetadot_model2[context:]], dim=1)
            state_pred_context = torch.stack([x_model2[number_of_context:], xdot_model2[number_of_context:], theta_model2[number_of_context:], thetadot_model2[number_of_context:]], dim=1)  # 7/26/2025
            state_rk4_context = torch.stack([xs_dataset[number_of_context:, 0], xs_dataset[number_of_context:, 1], xs_dataset[number_of_context:, 2], xs_dataset[number_of_context:, 3]], dim=1)
            mse_loss = mse(state_pred_context, state_rk4_context, device)
            print("mse_loss",mse_loss)
            mse_per_context.append(mse_loss)

            # phase_plot(theta_plot,thetadot_plot,theta_rk4_unscaled, save_results_path, folder_name)

            # mse_control_loss = mse_controls(controls_model2[start_index-1:], control_values_rk4[start_index-1:-1], device)
            mse_control_loss = mse_controls(controls_model2[number_of_context-1:], control_values_scaled[number_of_context-1:], device)
            print("mse_control_loss",mse_control_loss)
            mse_control_per_context.append(mse_control_loss)
                


            # for idx, context_length in enumerate(contexts):
            #     mse_results[context_length].append(mse_per_context[idx])
            #     mse_control_results[context_length].append(mse_control_per_context[idx])
        
            counter += 1
            
        # mse_mean = {context_length: np.mean(mse_results[context_length]) for context_length in contexts}
        # mse_std = {context_length: np.std(mse_results[context_length]) for context_length in contexts}
        # mse_control_mean = {context_length: np.mean(mse_control_results[context_length]) for context_length in contexts}
        # mse_control_std = {context_length: np.std(mse_control_results[context_length]) for context_length in contexts}
        # print(f"Mean MSE: {mse_mean}")
        # print(f"Std MSE: {mse_std}")

    # plot_mse_vs_context_length(mse_mean, mse_std, save_results_path, folder_name, mse_plot_label, loss_type="state")
    # plot_mse_vs_context_length(mse_control_mean, mse_control_std, save_results_path, folder_name, mse_plot_label, loss_type="control")


                

    return cartmasses, polemasses, polelenghs, phase_data, controls_data, data_and_controls, pends

try:
    # results = main()
    # model_checkpoint_step_list = np.arange(5000, 280001, 5000)
    # model_checkpoint_step_list = np.append(model_checkpoint_step_list, 284408)

    # model_checkpoint_step_list = np.arange(200,43601, 200)
    # model_checkpoint_step_list = np.append(model_checkpoint_step_list, 43734)
    # model_checkpoint_step_list =[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                                #  12000, 14000, 16000, 18000, 20000, 25000, 30000, 35000, 40000, 43734]

    # model_checkpoint_step_list = [8000, 9000, 10000, 
    #                               12000, 14000, 16000, 18000, 20000, 25000, 30000, 35000, 40000, 43734]
    # model_checkpoint_step_list = [16000, 20000, 30000, 40000, 43734]
    # model_checkpoint_step_list = [10000, 50000] #[17000]
    # model_checkpoint_step_list = [200, 400, 2000, 5000, 10000, 17000, 30000, 60000, 100000, 200000]
    model_checkpoint_step_list = [5000, 10000, 17000, 30000, 60000, 100000, 200000]
    Num_of_contexts = [1, 5, 10, 25, 50]
    # Num_of_contexts = [5]


    for step in tqdm(model_checkpoint_step_list, desc="Checkpoint Steps"):
        model_checkpoint_step = int(step)
        folder_name = f"inference_run/{plot_label}_{model_checkpoint_step}_{model_run_id}"

        phase_data = {context_length: [] for context_length in Num_of_contexts}
        controls_data = {context_length: [] for context_length in Num_of_contexts}
        data_and_controls = []

        for Num_of_context in Num_of_contexts:
            print(f"Running inference for checkpoint step {model_checkpoint_step} with context length {Num_of_context}...")
            
            results = main(model_checkpoint_step,
                            folder_name,
                            Num_of_context,
                            # mse_results,
                            # mse_control_results,
                            phase_data,
                            controls_data,
                            data_and_controls,                          
                           ) 
            _, _, _, phase_data, controls_data, data_and_controls, pends = results

        
        save_results = os.path.join(folder_name, f"results_maxcontext{Num_of_context}_numpends{Num_of_pendulums}_{mode}_alexcode.pkl")

        with open(save_results, "wb") as f:
            pickle.dump(results, f)


    print("done")   
except Exception:
    print("exception starting debugger")
    traceback.print_exc()
    ipdb.post_mortem()
