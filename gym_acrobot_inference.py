import os
import io
import gym
import math
import time
import imageio
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
# from gym_continuous_cartpole import ContinuousCartPoleEnv
# from gym_cartpole_swingup_lqr import swingup_lqr_controller
from gym_continuous_acrobot import AcrobotEnv

def wrap(x, m=-np.pi-0.002, M=np.pi-0.002):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def wrap_angle(angle):
    """Wraps an angle in radians to the range [-pi, pi].

    Args:
        angle: a scalar angle in radians
    Returns:
        wrapped_angle: the angle wrapped to the range [-pi, pi]
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _total_energy(theta1, theta2, d1, d2, m1, m2, l1, lc1, lc2, I1, I2, g=9.81):
    # small helper (matches your conventions)
    c2 = np.cos(theta2)
    D11 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2) + I1 + I2
    D12 = m2*(lc2**2 + l1*lc2*c2) + I2
    D22 = m2*lc2**2 + I2
    D = np.array([[D11, D12],[D12, D22]])
    dq = np.array([d1, d2])
    KE = 0.5 * dq @ (D @ dq)
    # T1 = 0.5 * I1 * d1**2
    # T2 = 0.5 * (m2 * l1**2 + I2 + 2*m2 * l1 * lc2 * c2) * d1**2 + 0.5 * I2 * d2**2 + (I2 + m2* l1 * lc2 * c2) * d1 * d2
    # KE = T1 + T2

    # U = -m1 * g * lc1 * np.cos(theta1 - np.pi/2.0) - m2 * g * (l1 * np.cos(theta1 - np.pi/2.0) + lc2 * np.cos(theta1 + theta2 - np.pi/2.0))
    # PE = U  # potential energy (up is negative; consistent with physics)

    # potential energy (up is positive; consistent with gym)
    y1 = lc1 * np.cos(theta1 - np.pi/2.0)
    y_tip = l1 * np.cos(theta1 - np.pi/2.0)
    y2 = y_tip + lc2 * np.cos(theta1 + theta2 - np.pi/2.0)
    PE = m1 * 9.81 * y1 + m2 * 9.81 * y2
    # return KE + PE
    return KE, PE


# def run_single_system(masscart, masspole, length, state_data, ctrl_data, run_idx, save_dir, context=None):
def run_single_system(link_length1, link_length2, link_mass1, link_mass2, state_data, ctrl_data, run_idx, save_dir, context=None, true_state_data=None, true_ctrl_data=None):
    link_com_pos_1 = 0.5 * link_length1
    link_com_pos_2 = 0.5 * link_length2
    link_moi_1 = (1/12) * link_mass1 * link_length1 **2
    link_moi_2 = (1/12) * link_mass2 * link_length2 **2
    env = AcrobotEnv(
        LINK_LENGTH_1=link_length1,
        LINK_LENGTH_2=link_length2,
        LINK_MASS_1=link_mass1,
        LINK_MASS_2=link_mass2,
        LINK_COM_POS_1=link_com_pos_1,
        LINK_COM_POS_2=link_com_pos_2,
        LINK_MOI1=link_moi_1,
        LINK_MOI2=link_moi_2,
        render_mode='rgb_array'
    )


    frames = []
    kinetic_energies = []
    potential_energies = []
    total_energies = []

    for i, s in enumerate(state_data):
        # env.state = np.array(s, dtype=np.float32)
        s = s.cpu().numpy() if isinstance(s, torch.Tensor) else np.array(s, dtype=np.float32)
        kin_engy, pot_engy = _total_energy(s[0], s[1], s[2], s[3], link_mass1, link_mass2,
                                          link_length1, link_com_pos_1, link_com_pos_2,
                                          link_moi_1, link_moi_2)
        kinetic_energies.append(kin_engy)
        potential_energies.append(pot_engy)
        total_energies.append(kin_engy + pot_engy)
        env.state = s
        frame = env.render()
        frames.append(frame)


    env.close()

    

    states = state_data.cpu() if isinstance(state_data, torch.Tensor) else state_data
    actions = ctrl_data.cpu() if isinstance(ctrl_data, torch.Tensor) else ctrl_data

    if true_state_data is not None and true_ctrl_data is not None:
        true_states = true_state_data.cpu() if isinstance(true_state_data, torch.Tensor) else true_state_data
        true_actions = true_ctrl_data.cpu() if isinstance(true_ctrl_data, torch.Tensor) else true_ctrl_data

    # print(f"actions shape: {actions.shape}, states shape: {states.shape}")
    # import pdb; pdb.set_trace()
    
    if context is None:
        # folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}"
        folder_name = f"l1_{link_length1:.2f}_l2_{link_length2:.2f}_m1_{link_mass1:.2f}_m2_{link_mass2:.2f}"
    else:
        # folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}_context_{context}"
        folder_name = f"l1_{link_length1:.2f}_l2_{link_length2:.2f}_m1_{link_mass1:.2f}_m2_{link_mass2:.2f}_context_{context}"
    path = os.path.join(save_dir, f"run_{run_idx:03}_{folder_name}")
    os.makedirs(path, exist_ok=True)

    # Save video
    imageio.mimsave(os.path.join(path, 'acrobot.mp4'), frames, fps=70)

    # Save control plot
    plt.figure(figsize=(13, 8))
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(actions[:, 0])), actions[:, 0], label='Control Actions', color='red', s=10)
    if true_ctrl_data is not None:
        plt.scatter(range(len(true_actions[:, 0])), true_actions[:, 0], label='Reference Control Actions', color='cyan', s=10, alpha=0.5)
    plt.title('Control Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(range(len(actions[:, 1])), actions[:, 1] + 1.0, label='Control Label', color='red', s=10)
    if true_ctrl_data is not None:
        plt.scatter(range(len(true_actions[:, 1])), true_actions[:, 1] + 1.0, label='Reference Control Label', color='cyan', s=10, alpha=0.5)
    plt.title('Control Labels Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Label')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(path, 'controls.png'))
    plt.savefig(os.path.join(path, 'controls.pdf'), format='pdf', bbox_inches='tight')
    plt.close()



    # Save state plots
    plt.figure(figsize=(15, 8))
    # labels = ['Cart Position (x)', 'Cart Velocity (x_dot)', 'Pole Angle (theta)', 'Pole Angular Velocity (theta_dot)']
    labels = ['Theta 1 (theta1)', 'Theta 2 (theta2)', 'Theta Dot 1 (dtheta1)', 'Theta Dot 2 (dtheta2)']
    # colors = ['red', 'green', 'orange', 'purple']
    colors = ['red', 'red', 'red', 'red']
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.scatter(range(len(states)), states[:, i], label="Model " + labels[i], color=colors[i], s=10)
        if true_state_data is not None:
            plt.scatter(range(len(true_states)), true_states[:, i], label="Reference " + labels[i], color='cyan', s=10, alpha=0.5)
        plt.title(labels[i] + ' Over Time') if true_state_data is None else plt.title(f'Model vs Reference {labels[i]} Over Time (Context: {context})')
        plt.xlabel('Time Step')
        plt.ylabel(labels[i].split('(')[-1].rstrip(')'))
        plt.grid()
        plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(path, 'states.png'))
    plt.savefig(os.path.join(path, 'states.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # # Save energy plots
    # plt.figure(figsize=(10, 7))
    # plt.plot(range(len(kinetic_energies)), kinetic_energies, label='Kinetic Energy', color='blue')
    # plt.plot(range(len(potential_energies)), potential_energies, label='Potential Energy', color='orange')
    # plt.plot(range(len(total_energies)), total_energies, label='Total Energy', color='green')
    # plt.title('Energy Over Time')
    # plt.xlabel('Time Step')
    # plt.ylabel('Energy')
    # plt.grid()
    # plt.legend()
    # plt.savefig(os.path.join(path, 'energies.png'))
    # plt.close()

    # Save States and controls on one plot
    # font parameters for plots
    parameters = {
        'font.size': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 16,
        'font.family': 'serif'
    }
    plt.rcParams.update(parameters)
    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(3, 2, i+1)
        # if i == 0:
        #     wrapped_states = np.array([wrap_angle(s[i]) for s in states])
        #     plt.scatter(range(len(states)), wrapped_states, label='Transformer', color=colors[i], s=10)
        # else:
        plt.scatter(range(len(states)), states[:, i], label='Transformer', color=colors[i], s=10)
        if true_state_data is not None and true_ctrl_data is not None:
            plt.scatter(range(len(true_states)), true_states[:, i], label='Reference', color='cyan', s=10, alpha=0.5)
        # plt.title(labels[i] + ' Over Time') if true_state_data is None else plt.title(f'Model vs Reference {labels[i]} Over Time (Context: {context})')
        # plt.xlabel('Time Step')
        plt.xlabel('Time Step', fontsize=24)
        # plt.ylabel(labels[i].split('(')[-1].rstrip(')'))
        plt.ylabel(labels[i].split('(')[-1].rstrip(')'), fontsize=24)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid()
        # plt.legend()
        # plt.legend(fontsize=16)

    plt.subplot(3, 2, 5)
    plt.scatter(range(len(actions)), actions[:,0], label='Transformer', color='red', s=10)
    if true_state_data is not None and true_ctrl_data is not None:
        plt.scatter(range(len(true_actions)), true_actions[:,0], label='Reference', color='cyan', s=10)
    # plt.title('Control Actions Over Time') if true_ctrl_data is None else plt.title(f'Predicted vs Reference Control Actions Over Time (Context: {context})')
    # plt.xlabel('Time Step')
    plt.xlabel('Time Step', fontsize=24)
    # plt.ylabel('Control Action')
    plt.ylabel('Control Action', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    # plt.legend()
    # plt.legend(fontsize=16)
    plt.subplot(3, 2, 6)
    plt.scatter(range(len(actions)), actions[:,1] + 1.0, label='Transformer', color='red', s=10)
    if true_state_data is not None and true_ctrl_data is not None:
        plt.scatter(range(len(true_actions)), true_actions[:,1] + 1.0, label='Reference', color='cyan', s=10, alpha=0.5)
    # plt.title('Control Labels Over Time') if true_ctrl_data is None else plt.title(f'Predicted vs Reference Control Labels Over Time (Context: {context})')
    # plt.xlabel('Time Step')
    plt.xlabel('Time Step', fontsize=24)
    # plt.ylabel('Control Label')
    plt.ylabel('Control Label', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    # plt.legend()
    # plt.legend(fontsize=16)

    # 1. Get handles and labels from the FIRST subplot (since they are all the same)
    # This is much cleaner than looping through all axes
    handles, labels = plt.gcf().axes[0].get_legend_handles_labels()

    # 2. Use fig.legend to attach it to the whole 20x10 canvas
    # We place it at the very top or very bottom of the FIGURE
    fig = plt.gcf()
    fig.legend(
        handles, 
        labels, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.02),
        ncol=2, 
        fontsize=24,
        markerscale=4.0,     # This multiplies the legend marker size by 4
        frameon=True, 
        edgecolor='black'
    )

    # 3. Adjust layout to leave room for the legend at the bottom
    # rect=[left, bottom, right, top]
    plt.tight_layout(rect=[0, 0.08, 1, 1])


    # plt.tight_layout()
    # plt.savefig(os.path.join(path, 'states_and_controls.png'))
    plt.savefig(os.path.join(path, 'states_and_controls.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    return {
        "path": path,
        # "masscart": masscart,
        # "masspole": masspole,
        # "length": length,
        "link_length1": link_length1,
        "link_length2": link_length2,
        "link_mass1": link_mass1,
        "link_mass2": link_mass2,
        # "theta_init": theta_init,
        # "thetadot_init": thetadot_init,
        # "stabilized": stabilized,
        "final_state": states[-1]
    }

def load_data(data_path):
    with open(data_path, 'rb') as f:
        # data = pickle.load(f)
            data = CPU_Unpickler(f).load()
        # data = torch.load(data_path, map_location=torch.device('cpu'))
        # b = f.read()
    
    # data = torch.load(io.BytesIO(b), map_location=torch.device('cpu'), weights_only=False)
    return data

import os
import pickle
import io
import os

# 1. Define a helper class to force CPU loading
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)




if __name__ == "__main__":
    # num_runs = 25
    results = []
    save_dir = os.path.join(os.getcwd(), 'videos', 'acrobot_inference_gym_runs')
    os.makedirs(save_dir, exist_ok=True)
    model_run_id = "efc700a2-0b51-4853-885d-557ba3c9d942" #"aa880853-841e-4b61-a7a7-9a3720482be2" #"e6ca8305-a383-4bc2-9f18-bd258dcc0183" #"15bf641c-dbc0-4f2f-b62f-fe04f568aacb" #"ec03ac2f-4708-4295-a44d-c14d439f7335" #"15bf641c-dbc0-4f2f-b62f-fe04f568aacb" #"d9d1d44a-9942-40b9-a2d4-bfba4177f2ce" ### finetuned lin layers chkpt #"15bf641c-dbc0-4f2f-b62f-fe04f568aacb" #"c953cb49-31b2-4829-8d1e-d9e2b1c99dce" #"056764e2-f56a-4e25-8019-3ce5098c388c" #"b3725997-9aee-4578-b668-d33e7cb29c4e" #"2ca9672c-582e-43ef-85cf-8550f325947a" #"a1d5f223-6768-4134-934b-4879031f7ea1" #"cf756e46-3ddb-4df7-9a13-ba850f495257" #"5b73d6c9-b526-4bfe-bab3-005f5369cf5a" #"a1d5f223-6768-4134-934b-4879031f7ea1" #"f8211c69-7ae8-47b3-9bd2-e<KEY>"#"de<KEY>" #"<KEY>"#"be<KEY>"#"eb<KEY>"#"a<KEY>"#"a<KEY>"#"<KEY>"#"<KEY>"#"be<KEY>"#"eb<KEY>"#"a<KEY>"#"a<KEY>"#"cf<KEY>"
    save_dir = os.path.join(save_dir, model_run_id)
    # os.makedirs(save_dir, exist_ok=True)
    step = 300000 #135000 #300000 #135000 #50000 #395000 #230000 #90000 #250000 #90000 #125000 #255000 #260000 #284408 # 237346 #207672 #185000 #80000 #105000 #50000 #55000#274941 #300800
    save_dir = os.path.join(save_dir, f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)
    numberpend = 6 #200 #5
    context = 100
    mode = 'indistr' # 'indistr, 'ood', 'train'
    data_path = f'inference_run/mse_control_{step}_{model_run_id}/results_maxcontext{context}_numpends{numberpend}_{mode}_alexcode.pkl'
    # cartmasses, polemasses, polelengths, phase_data, controls_data, data_and_controls, pends = load_data(data_path)
    link_lengths1, link_lengths2, link_masses1, link_masses2, phase_data, controls_data, data_and_controls, pends = load_data(data_path)
    # save_dir = os.path.join(save_dir, f"indistr")
    save_dir = os.path.join(save_dir, f"{mode}")
    os.makedirs(save_dir, exist_ok=True)


    # for cartpole_idx in range(len(cartmasses)):
    for acrobot_idx in range(len(link_lengths1)):
        if acrobot_idx == 1:
            # save_dir_inner = os.path.join(save_dir, f"run_{cartpole_idx:03}")
            save_dir_inner = os.path.join(save_dir, f"run_{acrobot_idx:03}")
            print(f"Running acrobot {acrobot_idx+1}/{len(link_lengths1)}")
            link_length1 = link_lengths1[acrobot_idx]
            link_length2 = link_lengths2[acrobot_idx]
            link_mass1 = link_masses1[acrobot_idx]
            link_mass2 = link_masses2[acrobot_idx]

            # ground_truth_data_controls = data_and_controls[0][acrobot_idx]
            # state_data = ground_truth_data_controls[0]
            # ctrl_data = ground_truth_data_controls[1]

            # run_single_system(link_mass1, link_mass2, link_length1, link_length2,
                            #   state_data, ctrl_data, acrobot_idx, save_dir_inner)
            for i, context in enumerate(phase_data.keys()):
                if context == 50:
                    ground_truth_data_controls = data_and_controls[0][acrobot_idx]
                    state_data = ground_truth_data_controls[0]
                    ctrl_data = ground_truth_data_controls[1]
                    context_state_data = phase_data[context][acrobot_idx]
                    context_ctrl_data = controls_data[context][acrobot_idx]
                    run_single_system(link_mass1, link_mass2, link_length1, link_length2,
                                    context_state_data, context_ctrl_data, acrobot_idx, save_dir_inner, context=context, true_state_data=state_data, true_ctrl_data=ctrl_data)
            
