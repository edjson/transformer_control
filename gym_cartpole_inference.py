import os
import gym
import math
import time
import imageio
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from gym_continuous_cartpole import ContinuousCartPoleEnv
from gym_cartpole_swingup_lqr import swingup_lqr_controller


def run_single_system(masscart, masspole, length, state_data, ctrl_data, run_idx, save_dir, context=None, true_state_data=None, true_ctrl_data=None):
    env = ContinuousCartPoleEnv(
        masscart=masscart,
        masspole=masspole,
        length=length,
        render_mode="rgb_array"
    )

    # import pdb; pdb.set_trace()
    # state_data = state_data.cpu().numpy()
    # ctrl_data = ctrl_data.cpu().numpy()

    # obs, _ = env.reset(options={"init_state": [0.0, 0.0, theta_init, thetadot_init]})
    # frames, states, actions = [], [obs], []
    # switched = False
    # max_steps = 560

    # for _ in range(max_steps):
    #     action, switched = swingup_lqr_controller(obs, switched, masscart, masspole, length)
    #     obs, reward, done, truncated, _, applied_action = env.step(action)
    #     frame = env.render()
    #     frames.append(frame)
    #     states.append(obs)
    #     actions.append(applied_action)
    #     if done or truncated:
    #         break

    frames = []

    for i, s in enumerate(state_data):
        # env.state = np.array(s, dtype=np.float32)
        env.state = s
        frame = env.render()
        frames.append(frame)


    env.close()

    # Final state
    # states = np.array(states)
    # actions = np.array(actions).squeeze()

    # states = np.array(state_data)
    # actions = np.array(ctrl_data).squeeze()

    states = state_data.cpu() if isinstance(state_data, torch.Tensor) else state_data
    actions = ctrl_data.cpu() if isinstance(ctrl_data, torch.Tensor) else ctrl_data

    if true_state_data is not None and true_ctrl_data is not None:
        true_states = true_state_data.cpu() if isinstance(true_state_data, torch.Tensor) else true_state_data
        true_actions = true_ctrl_data.cpu() if isinstance(true_ctrl_data, torch.Tensor) else true_ctrl_data

    # states = torch.tensor(state_data, dtype=torch.float32)
    # actions = torch.tensor(ctrl_data, dtype=torch.float32)
    # final_theta = states[-1, 2]
    # final_theta_dot = states[-1, 3]
    # stabilized = abs((final_theta + np.pi) % (2*np.pi) - np.pi) < 0.2 and abs(final_theta_dot) < 0.5
    # theta_wrapped = (final_theta + np.pi) % (2 * np.pi) - np.pi
    # stabilized = abs(theta_wrapped) < 0.2 and abs(final_theta_dot) < 0.5
    # Output path and naming
    if context is None:
        folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}"
    else:
        folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}_context_{context}"
    path = os.path.join(save_dir, f"run_{run_idx:03}_{folder_name}")
    os.makedirs(path, exist_ok=True)

    # Save video
    imageio.mimsave(os.path.join(path, 'cartpole.mp4'), frames, fps=50)

    # Save control plot
    plt.figure(figsize=(13, 8))
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(actions)), actions[:,0], label='Predicted Control Actions', color='red', s=10)
    if true_state_data is not None and true_ctrl_data is not None:
        plt.scatter(range(len(true_actions)), true_actions[:,0], label='Reference Control Actions', color='cyan', s=10)
    plt.title('Control Actions Over Time') if true_ctrl_data is None else plt.title(f'Predicted vs Reference Control Actions Over Time (Context: {context})')
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(actions)), actions[:,1] + 1.0, label='Predicted Control Labels', color='red', s=10)
    if true_state_data is not None and true_ctrl_data is not None:
        updated_true_actions = true_actions[:,1] + 1.0  # Shift true control labels up by 1.0 for better visualization
        plt.scatter(range(len(true_actions)), updated_true_actions, label='Reference Control Labels', color='cyan', s=10, alpha=0.5)
    plt.title('Control Labels Over Time') if true_ctrl_data is None else plt.title(f'Predicted vs Reference Control Labels Over Time (Context: {context})')
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
    labels = ['Cart Position (x)', 'Cart Velocity (x_dot)', 'Pole Angle (theta)', 'Pole Angular Velocity (theta_dot)']
    # colors = ['red', 'green', 'orange', 'purple']
    colors = ['red', 'red', 'red', 'red']
    for i in range(4):
        plt.subplot(2, 2, i+1)
        # wrap angle for better visualization
        # if i == 2:
        #     wrapped_states = (states[:, i] + np.pi) % (2 * np.pi) - np.pi
        #     plt.scatter(range(len(states)), wrapped_states, label='Model ' + labels[i], color=colors[i], s=10)
        #     if true_state_data is not None and true_ctrl_data is not None:
        #         wrapped_true_states = (true_states[:, i] + np.pi) % (2 * np.pi) - np.pi
        #         plt.scatter(range(len(true_states)), wrapped_true_states, label='True ' + labels[i], color='cyan', s=10)

        # else:
        plt.scatter(range(len(states)), states[:, i], label='Model ' + labels[i], color=colors[i], s=10)
        if true_state_data is not None and true_ctrl_data is not None:
            plt.scatter(range(len(true_states)), true_states[:, i], label='Reference ' + labels[i], color='cyan', s=10, alpha=0.5)
        plt.title(labels[i] + ' Over Time') if true_state_data is None else plt.title(f'Model vs Reference {labels[i]} Over Time (Context: {context})')
        plt.xlabel('Time Step')
        plt.ylabel(labels[i].split('(')[-1].rstrip(')'))
        plt.grid()
        plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(path, 'states.png'))
    plt.savefig(os.path.join(path, 'states.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

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
        # plt.subplot(2, 3, i+1)
        plt.scatter(range(len(states)), states[:, i], label='Transformer', color=colors[i], s=10)
        if true_state_data is not None and true_ctrl_data is not None:
            plt.scatter(range(len(true_states)), true_states[:, i], label='Reference', color='cyan', s=10, alpha=0.5)
        # plt.title(labels[i] + ' Over Time') if true_state_data is None else plt.title(f'Transformer vs Reference {labels[i]} Over Time (Context: {context})')
        # plt.xlabel('Time Step', fontsize)
        plt.xlabel('Time Step', fontsize=24)
        # plt.ylabel(labels[i].split('(')[-1].rstrip(')'))
        plt.ylabel(labels[i].split('(')[-1].rstrip(')'), fontsize=24)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid()
        # plt.legend()
        # plt.legend(fontsize=16)

    plt.subplot(3, 2, 5)
    # plt.subplot(2, 3, 5)
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
    # plt.subplot(2, 3, 6)
    plt.scatter(range(len(actions)), actions[:,1] + 1.0, label='Transformer', color='red', s=10)
    if true_state_data is not None and true_ctrl_data is not None:
        updated_true_actions = true_actions[:,1] + 1.0  # Shift true control labels up by 1.0 for better visualization
        plt.scatter(range(len(true_actions)), updated_true_actions, label='Reference', color='cyan', s=10, alpha=0.5)
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
    # one big legend for the whole figure

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

    # handles, labels = [], []
    # for ax in plt.gcf().axes:
    #     h, l = ax.get_legend_handles_labels()
    #     # print(f"h: {h}, l: {l}")
    #     # if l not in labels:  # Avoid duplicate labels in the legend
    #     handles.extend(h)
    #     labels.extend(l)
    #     if labels == l:
    #         break
    # plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True, shadow=False, framealpha=0.6, fontsize=16)
    # plt.tight_layout()
    # plt.savefig(os.path.join(path, 'states_and_controls.png'))
    plt.savefig(os.path.join(path, 'states_and_controls.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    return {
        "path": path,
        "masscart": masscart,
        "masspole": masspole,
        "length": length,
        # "theta_init": theta_init,
        # "thetadot_init": thetadot_init,
        # "stabilized": stabilized,
        "final_state": states[-1]
    }

def load_data(data_path):
    with open(data_path, 'rb') as f:
        # data = pickle.load(f)
        data = CPU_Unpickler(f).load()
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
    save_dir = os.path.join(os.getcwd(), 'videos', 'cartpole_inference_gym_runs')
    # os.makedirs(save_dir, exist_ok=True)
    model_run_id = "fc5cb0fe-d4c6-4fba-8196-01a37d611194"#"5a28bfa8-c81f-473b-996f-879dfadc62df"#"3f8379cf-1cf0-47db-9210-5abb1c43e5fb" #"b3725997-9aee-4578-b668-d33e7cb29c4e" #"2ca9672c-582e-43ef-85cf-8550f325947a" #"a1d5f223-6768-4134-934b-4879031f7ea1" #"cf756e46-3ddb-4df7-9a13-ba850f495257" #"5b73d6c9-b526-4bfe-bab3-005f5369cf5a" #"a1d5f223-6768-4134-934b-4879031f7ea1" #"f8211c69-7ae8-47b3-9bd2-e4f0edda1e82"#"de00c432-078f-44d1-8cc9-e6ab0dfdca88" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"c3af70ca-3733-4cec-a876-95db9bc9a593" #"32ea0675-5539-4d02-80fb-7bfe1f4c263e" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"57f687d9-9e41-48f5-83d4-559592ca762b" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"be268b82-d25e-4026-b303-91ba3c6d1e9f" #"eb14c6a6-d8eb-4b1c-8f37-a57c04a2d66b" #"a49e6137-5856-47ca-86d8-1d37f2f11e7b" #"a885c11b-dee1-472b-9577-46c8c03f70b3" #"9329819d-2d02-4ff0-8741-ed2959c98fdd" #"850d5e45-7a37-4e6d-8499-ac7b11076de3" #"0215ad56-8bf1-47e0-b509-db5af831dabe" #"ee0f5a22-9607-43e5-8cd9-db2f1be66a23"
    save_dir = os.path.join(save_dir, model_run_id)
    # os.makedirs(save_dir, exist_ok=True)
    step = 10000 #50000 #17000 #30000 #55000#274941 #300800
    save_dir = os.path.join(save_dir, f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)
    numberpend = 5 #3 #11 #10 #200 #5
    context = 50
    mode = 'indistr' # 'train', 'ood', 'indistr'
    case_type = 'passing_cases' # 'passing_cases', 'failure_cases'
    data_path = f'inference_run/mse_control_{step}_{model_run_id}/results_maxcontext{context}_numpends{numberpend}_{mode}_alexcode.pkl'
    cartmasses, polemasses, polelengths, phase_data, controls_data, data_and_controls, pends = load_data(data_path)
    save_dir = os.path.join(save_dir, f"{mode}_cartpoles_{case_type}")
    os.makedirs(save_dir, exist_ok=True)

    # counter = 0
    for cartpole_idx in range(len(cartmasses)):
        save_dir_inner = os.path.join(save_dir, f"run_{cartpole_idx:03}")
        print(f"Running cartpole {cartpole_idx+1}/{len(cartmasses)}")
        masscart = cartmasses[cartpole_idx]
        masspole = polemasses[cartpole_idx]
        length = polelengths[cartpole_idx]

        # ground_truth_data_controls = data_and_controls[0][cartpole_idx]
        # state_data = ground_truth_data_controls[0]
        # ctrl_data = ground_truth_data_controls[1] #[:, 0]

        # run_single_system(masscart, masspole, length, 
                        #   state_data, ctrl_data, cartpole_idx, save_dir_inner)
        for i, context in enumerate(phase_data.keys()):
            # print(f"len(phase_data[{context}]): {len(phase_data[context])}")
            ground_truth_data_controls = data_and_controls[i][cartpole_idx]
            state_data = ground_truth_data_controls[0]
            ctrl_data = ground_truth_data_controls[1] #[:, 0]
            context_state_data = phase_data[context][cartpole_idx]
            context_ctrl_data = controls_data[context][cartpole_idx] #[:, 0]
            run_single_system(masscart, masspole, length, 
                              context_state_data, context_ctrl_data, cartpole_idx, save_dir_inner, context=context, true_state_data=state_data, true_ctrl_data=ctrl_data)
            
