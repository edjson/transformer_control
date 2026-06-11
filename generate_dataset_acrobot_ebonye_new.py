import os
import random
from tqdm import tqdm
from samplers import PendulumSampler
from samplers import CartPoleSampler
from samplers import CartPoleSampler2
from samplers import AcrobotSampler
from curriculum import Curriculum
from random import randint
import uuid
import ipdb

from quinine import QuinineArgumentParser
import torch
import yaml
from schema import schema
from models import build_model
import math
import random
import numpy as np
import torch
import pickle
import crocoddyl

# seed = [1]
torch.backends.cudnn.benchmark = True

def get_valid_masses_and_lengths_uniform(size=1, LINK_LENGTH_1_lowerbound=0.5, LINK_LENGTH_1_upperbound=1.0,
                                         LINK_LENGTH_2_lowerbound=1.0, LINK_LENGTH_2_upperbound=1.5,
                                         LINK_MASS_1_lowerbound=1.5, LINK_MASS_1_upperbound=2.0,
                                         LINK_MASS_2_lowerbound=1.0, LINK_MASS_2_upperbound=1.5):
    """
    Samples valid masses and lengths for a pendulum system that meet specific constraints, using a uniform distribution.

    Args:
        LINK_LENGTH_1_lowerbound (float): The lower bound for the first link length.
        LINK_LENGTH_1_upperbound (float): The upper bound for the first link length.
        LINK_LENGTH_2_lowerbound (float): The lower bound for the second link length.
        LINK_LENGTH_2_upperbound (float): The upper bound for the second link length.
        LINK_MASS_1_lowerbound (float): The lower bound for the first link mass.
        LINK_MASS_1_upperbound (float): The upper bound for the first link mass.
        LINK_MASS_2_lowerbound (float): The lower bound for the second link mass.
        LINK_MASS_2_upperbound (float): The upper bound for the second link mass.

    Returns:
        tuple: A tuple containing:
           - LINK_LENGTH_1 (float): The sampled first link length.
           - LINK_LENGTH_2 (float): The sampled second link length.
           - LINK_MASS_1 (float): The sampled first link mass.
           - LINK_MASS_2 (float): The sampled second link mass.
    """
    LINK_LENGTH_1 = sample_uniform(LINK_LENGTH_1_lowerbound, LINK_LENGTH_1_upperbound, size=size)
    LINK_LENGTH_2 = sample_uniform(LINK_LENGTH_2_lowerbound, LINK_LENGTH_2_upperbound, size=size)
    LINK_MASS_1 = sample_uniform(LINK_MASS_1_lowerbound, LINK_MASS_1_upperbound, size=size)
    LINK_MASS_2 = sample_uniform(LINK_MASS_2_lowerbound, LINK_MASS_2_upperbound, size=size)
    return LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2

def sample_uniform(lower_bound, upper_bound, size=1):
    """
    Samples a value from a uniform distribution within specified bounds.

    Args:
        lower_bound (float): The lower bound of the sampled value.
        upper_bound (float): The upper bound of the sampled value.
        size (int, optional): The number of samples to draw. Defaults to 1.

    Returns:
        float or np.ndarray: A sampled value or array of values within the specified bounds.
    """
    return np.random.uniform(lower_bound, upper_bound, size=size)

def extract_window_fast(xs, ys, true_xs, window_size=800):
    # before = window_size - 300
    before = window_size - 450
    
    cos_theta1s = torch.cos(xs[:, :, 0])
    
    new_xs_list = []
    new_ys_list = []
    idxes = []
    batch_idx_list = []
    new_true_xs_list = []

    for i in range(xs.shape[0]):
        cos_theta1 = cos_theta1s[i]
        avg_rev = torch.cumsum(cos_theta1.flip(0), dim=0) / torch.arange(1, len(cos_theta1)+1, device=cos_theta1.device)
        avg_cumsum_rev = avg_rev.flip(0)
        indices = torch.where(avg_cumsum_rev < -0.96)[0]

        if len(indices) > 0:
            idx = indices[0].item()
            idxes.append(idx)
            batch_idx_list.append(i)  # keep track of original batch index

            if idx + window_size <= xs.shape[1]:
                start = max(0, idx - before)
                end = start + window_size
                new_xs_list.append(xs[i, start:end])
                new_ys_list.append(ys[i, start:end])
                new_true_xs_list.append(true_xs[i, start:end])
            else:
                new_xs_list.append(xs[i, -window_size:])
                new_ys_list.append(ys[i, -window_size:])
                new_true_xs_list.append(true_xs[i, -window_size:])
        # else: automatically skip trajectory

    if len(new_xs_list) == 0:
        return torch.empty(0, window_size, xs.shape[2], device=xs.device), \
               torch.empty(0, window_size, ys.shape[2], device=ys.device), \
               torch.empty(0, window_size, true_xs.shape[2], device=true_xs.device), \
               [], []

    new_xs = torch.stack(new_xs_list)
    new_ys = torch.stack(new_ys_list)
    new_true_xs = torch.stack(new_true_xs_list)

    # return new_xs, new_ys, batch_idx_list, idxes
    return new_xs, new_ys, new_true_xs, batch_idx_list, idxes

def extract_window_curated(xs, ys, true_xs, window_size=120):
    # before = window_size - 300
    # before = window_size - 450
    """"
    Curates windows based on three distint phase focuses:
    1. System ID (passive u=0)
    2. Transition (swing-up to LQR)
    3. Terminal (Steady-state stability)
    """
    # before = int(0.33 * window_size)
    
    cos_theta1s = torch.cos(xs[:, :, 0])
    total_len = xs.shape[1]
    
    new_xs_list = []
    new_ys_list = []
    idxes = []
    batch_idx_list = []
    new_true_xs_list = []

    for i in range(xs.shape[0]):
        # 1. find the transition point
        cos_theta1 = cos_theta1s[i]
        avg_rev = torch.cumsum(cos_theta1.flip(0), dim=0) / torch.arange(1, len(cos_theta1)+1, device=cos_theta1.device)
        avg_cumsum_rev = avg_rev.flip(0)
        indices = torch.where(avg_cumsum_rev < -0.96)[0]

        # t_switch = indices[0].item() if len(indices) > 0 else total_len // 2
        # if t_switch == indices[0].item():
        #     idxes.append(t_switch)
        #     batch_idx_list.append(i)  # keep track of original batch index
        if len(indices) > 0:
            t_switch = indices[0].item()
            idxes.append(t_switch)
            batch_idx_list.append(i)  # keep track of original batch index

            # 2. Decide which bucket to sample for this trajectory slice
            r = torch.rand(1).item()
            if r < 0.33:
                # BUCKET 1: System ID (passive u=0)
                start = 0
            elif r < 0.66:
                # BUCKET 2: Terminal LQR focus
                lower_bound = t_switch
                upper_bound = total_len - window_size - 1

                if upper_bound > lower_bound:
                    start = torch.randint(lower_bound, upper_bound, (1,)).item()
                else:
                    start = max(0, t_switch - window_size -1)
            else:
                # BUCKET 3: The transition region around the switch point
                offset = int(0.33 * window_size)
                start = max(0, t_switch - offset)
                start = min(start, total_len - window_size - 1)

            # 3. Extract the window
            end = start + window_size
            new_xs_list.append(xs[i, start:end])
            new_ys_list.append(ys[i, start:end])
            new_true_xs_list.append(true_xs[i, start:end])

    new_xs = torch.stack(new_xs_list)
    new_ys = torch.stack(new_ys_list)
    new_true_xs = torch.stack(new_true_xs_list)

    # return new_xs, new_ys, batch_idx_list, idxes
    return new_xs, new_ys, new_true_xs, batch_idx_list, idxes

# def prefilter_trajs(xs, ys, min_control_sum=200):
def prefilter_trajs(xs, ys, true_xs, min_control_sum=200):
    """
    Prefilters trajectories based on a threshold on control labels sum.

    Args:
        xs (torch.Tensor): The state trajectories of shape (batch_size, time_steps, state_dim).
        ys (torch.Tensor): The control input trajectories of shape (batch_size, time_steps, control_dim).
        cos_threshold (float): The cosine threshold for filtering.
        min_control_sum (float): The minimum sum of control inputs for filtering.
    Returns:
        filtered_xs (torch.Tensor): The filtered state trajectories.
        filtered_ys (torch.Tensor): The filtered control input trajectories.
        filtered_true_xs (torch.Tensor): The filtered true state trajectories.
        indices (list): The indices of the trajectories that passed the filtering.       
        
        
    """
    filtered_xs = []
    filtered_ys = []
    filtered_true_xs = []
    indices = []
    for i in range(xs.shape[0]):
        if torch.sum(ys[i, -500:, 1]) > min_control_sum:
            filtered_xs.append(xs[i])
            filtered_ys.append(ys[i])
            filtered_true_xs.append(true_xs[i])
            indices.append(i)
    if len(filtered_xs) == 0:
        return torch.empty(0, xs.shape[1], xs.shape[2], device=xs.device), \
               torch.empty(0, ys.shape[1], ys.shape[2], device=ys.device), \
               torch.empty(0, true_xs.shape[1], true_xs.shape[2], device=true_xs.device), \
               []
    return torch.stack(filtered_xs), torch.stack(filtered_ys), torch.stack(filtered_true_xs), indices




def save_pickle(data, pickle_path):
    """
    Saves data to a pickle file.

    Args:
        data (any): The data to save.
        pickle_path (str): The path to the pickle file where the data will be saved.
    """
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)


def append_to_dataset_logger(iteration, LINK_MASS_1, LINK_MASS_2, LINK_LENGTH_1, LINK_LENGTH_2, xs_shape, log_file):
    """
    Appends simulation details to a seed file for tracking and reproducibility.
    
    Args:
        iteration (int): The iteration number of the simulation.
        cartmass (float): The mass of the cart used in the simulation.
        polemass (float): The mass of the pole used in the simulation.
        polelength (float): The length of the pole used in the simulation.
        xs_shape (tuple): The shape of the state dataset (batch_size, n_dim, timepoints).
        log_file (str): The path to the log file.
    """
    with open(log_file, 'a') as f:
        f.write(f"Iteration: {iteration}\n")
        f.write("LINK_MASS_1: " + str(LINK_MASS_1) + "\n")
        f.write("LINK_MASS_2: " + str(LINK_MASS_2) + "\n")
        f.write("LINK_LENGTH_1: " + str(LINK_LENGTH_1) + "\n")
        f.write("LINK_LENGTH_2: " + str(LINK_LENGTH_2) + "\n")
        f.write(f"xs.shape: {xs_shape}\n")
        f.write("-" * 10 + "\n")


def make_train_data(args):
    """
    Geneates training datasets for an inverted pendulum system simulation and saves them 
    as pickle files along with metadata for reproducibility.

    Args:
        args (Namespace):
            - args.training.train_steps (int): The total number of pickle files.
            - args.training.batch_size (int): batch of each pickle file.
            - args.training.curriculum (dict): Curriculum settings to adjust training parameters dynamically.
            - args.dataset_filesfolder (str): The directory where dataset files and logs are stored.
            - args.dataset_logger_textfile (str): The name of the file for logging dataset info.
            - args.pickle_folder (str): The dataset_filesfolder subfolder where generated dataset pickle files are saved.

    Notes:
        - The `PendulumSampler` is used to generate the dataset based on random valid pendulum parameters (masses and lengths).
        - The generated datasets are saved as pickle files named in the format `multipendulum_{i}.pkl`.
        - Metadata such as the current seed, pendulum parameters, and dataset shape is logged in a separate file for reproducibility.
        - The curriculum dynamically updates training parameters, such as the number of points in each dataset.

    """
    curriculum = Curriculum(args.training.curriculum)
    starting_step = 6000
    # starting_step = 0
    bsize = args.training.batch_size
    pbar = tqdm(range(starting_step, args.training.train_steps + args.training.test_pendulums + args.training.test_pendulums_outofdistr)) 
    dt = 0.02  # Time step for simulation
    # pbar_test = tqdm(range(args.training.test_pendulums))
    # num_test_pendulums = args.training.test_pendulums

    # seed_file = os.path.join(args.dataset_filesfolder, args.dataset_logger_textfile)
    train_logger = os.path.join(args.dataset_filesfolder, args.dataset_logger_textfile)
    test_logger = os.path.join(args.dataset_filesfolder, args.dataset_test_logger_textfile)
    test_logger_outofdistr = os.path.join(args.dataset_filesfolder, args.dataset_test_outofdistr_logger_textfile) ## 3/5/2025 out of distribution data
    base_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder)
    test_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder_test)
    test_data_dir_outofdistr = os.path.join(args.dataset_filesfolder, args.pickle_folder_test_outofdistr) ## 3/5/2025 out of distribution data
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(test_data_dir_outofdistr, exist_ok=True) ## 3/5/2025 out of distribution data

    contexts = [1, 5, 10, 25, 50, 75, 100]
   
    for i in pbar:
          
        if i < args.training.train_steps:
            LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2 = get_valid_masses_and_lengths_uniform(size=bsize, LINK_LENGTH_1_lowerbound=0.5, LINK_LENGTH_1_upperbound=1.0,
                                                                                                         LINK_LENGTH_2_lowerbound=1.0, LINK_LENGTH_2_upperbound=1.5,
                                                                                                         LINK_MASS_1_lowerbound=1.5, LINK_MASS_1_upperbound=2.0,
                                                                                                         LINK_MASS_2_lowerbound=1.0, LINK_MASS_2_upperbound=1.5)
            sampler = AcrobotSampler(n_dims=4)
            T, xs, control_values, true_xs = sampler.generate_xs_dataset(curriculum.n_points, b_size=bsize,
                                                                         LINK_LENGTH_1=LINK_LENGTH_1,
                                                                         LINK_LENGTH_2=LINK_LENGTH_2,
                                                                         LINK_MASS_1=LINK_MASS_1,
                                                                         LINK_MASS_2=LINK_MASS_2,
                                                                         dt=dt, #test_mode=False
                                                                         test_mode={'on': False, 'context': 100}
                                                                         ) 

            # T_cpu = T.cpu()
            xs_cpu = xs.cpu()
            control_values_cpu = control_values.cpu()
            true_xs_cpu = true_xs.cpu()

            pickle_file = f'batch_{i}.pkl'
            pickle_path = os.path.join(base_data_dir, pickle_file)

            # ## prune sequences that do not stabilize
            # xs_cpu, control_values_cpu, true_xs_cpu, indices = prefilter_trajs(xs_cpu, control_values_cpu, true_xs_cpu, min_control_sum=200)
            # # T_cpu = T_cpu[indices]
            # # import pdb; pdb.set_trace()
            # LINK_LENGTH_1 = LINK_LENGTH_1[indices]
            # LINK_LENGTH_2 = LINK_LENGTH_2[indices]
            # LINK_MASS_1 = LINK_MASS_1[indices]
            # LINK_MASS_2 = LINK_MASS_2[indices]

            ## extract windows around stabilization events
            # xs_cpu, control_values_cpu, batch_idxes, idxes = extract_window_fast(xs_cpu, control_values_cpu, window_size=800)
            # xs_cpu, control_values_cpu, batch_idxes, idxes = extract_window_fast(xs_cpu, control_values_cpu, window_size=600)
            # xs_cpu, control_values_cpu, true_xs_cpu, batch_idxes, idxes = extract_window_fast(xs_cpu, control_values_cpu, true_xs_cpu, window_size=600)
            xs_cpu, control_values_cpu, true_xs_cpu, batch_idxes, idxes = extract_window_curated(xs_cpu, control_values_cpu, true_xs_cpu, window_size=120)
            # T_cpu = T_cpu[batch_idxes]
            LINK_LENGTH_1 = LINK_LENGTH_1[batch_idxes]
            LINK_LENGTH_2 = LINK_LENGTH_2[batch_idxes]
            LINK_MASS_1 = LINK_MASS_1[batch_idxes]
            LINK_MASS_2 = LINK_MASS_2[batch_idxes]


            
            

            # save_pickle((xs, control_values, cartmass, polemass, polelength), pickle_path)
            # append_to_dataset_logger(i, cartmass, polemass, polelength, xs.shape, train_logger)

            save_pickle((xs_cpu, control_values_cpu, LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2, true_xs_cpu), pickle_path)
            append_to_dataset_logger(i, LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2, xs_cpu.shape, train_logger)

            # free gpu memory
            del T, xs, control_values, LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2, control_values_cpu, xs_cpu, true_xs_cpu
            # torch.cuda.empty_cache()


        elif i >= args.training.train_steps and i < args.training.train_steps + args.training.test_pendulums:
            test_batch_size = 3000
            LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2 = get_valid_masses_and_lengths_uniform(size=test_batch_size, LINK_LENGTH_1_lowerbound=0.5, LINK_LENGTH_1_upperbound=1.0,
                                                                                                         LINK_LENGTH_2_lowerbound=1.0, LINK_LENGTH_2_upperbound=1.5,
                                                                                                         LINK_MASS_1_lowerbound=1.5, LINK_MASS_1_upperbound=2.0,
                                                                                                         LINK_MASS_2_lowerbound=1.0, LINK_MASS_2_upperbound=1.5)
            sampler = AcrobotSampler(n_dims=4)
            for context in contexts:

                T, xs, control_values, true_xs = sampler.generate_xs_dataset(curriculum.n_points, b_size=test_batch_size,
                                                                            LINK_LENGTH_1=LINK_LENGTH_1,
                                                                            LINK_LENGTH_2=LINK_LENGTH_2,
                                                                            LINK_MASS_1=LINK_MASS_1,
                                                                            LINK_MASS_2=LINK_MASS_2,
                                                                            dt=dt,# test_mode=True
                                                                            test_mode={'on': True, 'context': context}
                                                                            ) 

                # T_cpu = T.cpu()
                xs_cpu = xs.cpu()
                control_values_cpu = control_values.cpu()
                true_xs_cpu = true_xs.cpu()
                
                pickle_file = f'batch_test_{i-args.training.train_steps}_{context}.pkl'
                pickle_path = os.path.join(test_data_dir, pickle_file)

                ## prune sequences that do not stabilize
                # xs_cpu, control_values_cpu, indices = prefilter_trajs(xs_cpu, control_values_cpu, min_control_sum=200)
                xs_cpu, control_values_cpu, true_xs_cpu, indices = prefilter_trajs(xs_cpu, control_values_cpu, true_xs_cpu, min_control_sum=200)
                # T_cpu = T_cpu[indices]
                LINK_LENGTH_1_new = LINK_LENGTH_1[indices]
                LINK_LENGTH_2_new = LINK_LENGTH_2[indices]
                LINK_MASS_1_new = LINK_MASS_1[indices]
                LINK_MASS_2_new = LINK_MASS_2[indices]

                ## extract windows around stabilization events
                # xs_cpu, control_values_cpu, batch_idxes, idxes = extract_window_fast(xs_cpu, control_values_cpu, window_size=800)
                # xs_cpu, control_values_cpu, batch_idxes, idxes = extract_window_fast(xs_cpu, control_values_cpu, window_size=600)
                # xs_cpu, control_values_cpu, true_xs_cpu, batch_idxes, idxes = extract_window_fast(xs_cpu, control_values_cpu, true_xs_cpu, window_size=600)
                # # T_cpu = T_cpu[batch_idxes]
                # LINK_LENGTH_1 = LINK_LENGTH_1[batch_idxes]
                # LINK_LENGTH_2 = LINK_LENGTH_2[batch_idxes]
                # LINK_MASS_1 = LINK_MASS_1[batch_idxes]
                # LINK_MASS_2 = LINK_MASS_2[batch_idxes]
                



                save_pickle((xs_cpu, control_values_cpu, LINK_LENGTH_1_new, LINK_LENGTH_2_new, LINK_MASS_1_new, LINK_MASS_2_new, true_xs_cpu), pickle_path)
                append_to_dataset_logger(i-args.training.train_steps, LINK_LENGTH_1_new, LINK_LENGTH_2_new, LINK_MASS_1_new, LINK_MASS_2_new, xs_cpu.shape, test_logger)

                # free gpu memory
                del T, xs, control_values, LINK_LENGTH_1_new, LINK_LENGTH_2_new, LINK_MASS_1_new, LINK_MASS_2_new, control_values_cpu, xs_cpu, true_xs_cpu
                # torch.cuda.empty_cache()
        else:
            # sampler = CartPoleSampler(n_dims=4)
            # cartmass, polemass, polelength = get_valid_masses_and_lengths_uniform(cartmasslowerbound=2, cartmassupperbound=3, polemasslowerbound=1.6, polemassupperbound=2.6, polelengthlowerbound=1.6, polelengthupperbound=2.6) ### 4/9/2025 (ebonye) cartpole system
            # T, xs, control_values = sampler.generate_xs_dataset(curriculum.n_points, cartmass=cartmass, polemass=polemass, polelength=polelength) ### 4/9/2025 (ebonye) cartpole system
            # pickle_file = f'multipendulum_test_outofdistr_{i-args.training.train_steps-args.training.test_pendulums}.pkl'
            # pickle_path = os.path.join(test_data_dir_outofdistr, pickle_file)
            # save_pickle((xs, control_values, cartmass, polemass, polelength), pickle_path)
            # append_to_dataset_logger(i-args.training.train_steps-args.training.test_pendulums, cartmass, polemass, polelength, xs.shape, test_logger_outofdistr)
            # pass
            test_batch_size = 3000
            LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2 = get_valid_masses_and_lengths_uniform(size=test_batch_size, LINK_LENGTH_1_lowerbound=1.0, LINK_LENGTH_1_upperbound=1.5,
                                                                                                         LINK_LENGTH_2_lowerbound=1.5, LINK_LENGTH_2_upperbound=2.0,
                                                                                                        #  LINK_MASS_1_lowerbound=1.5, LINK_MASS_1_upperbound=2.0,
                                                                                                            LINK_MASS_1_lowerbound=1.0, LINK_MASS_1_upperbound=1.5,
                                                                                                         LINK_MASS_2_lowerbound=1.0, LINK_MASS_2_upperbound=1.5)
            sampler = AcrobotSampler(n_dims=4)
            for context in contexts:
                T, xs, control_values, true_xs = sampler.generate_xs_dataset(curriculum.n_points, b_size=test_batch_size,
                                                                            LINK_LENGTH_1=LINK_LENGTH_1,
                                                                            LINK_LENGTH_2=LINK_LENGTH_2,
                                                                            LINK_MASS_1=LINK_MASS_1,
                                                                            LINK_MASS_2=LINK_MASS_2,
                                                                            dt=dt, #test_mode=True
                                                                            test_mode={'on': True, 'context': context}
                                                                            ) 

                # T_cpu = T.cpu()
                xs_cpu = xs.cpu()
                control_values_cpu = control_values.cpu()
                true_xs_cpu = true_xs.cpu()
                
                pickle_file = f'batch_test_{i-args.training.train_steps-args.training.test_pendulums}_{context}.pkl'
                pickle_path = os.path.join(test_data_dir_outofdistr, pickle_file)

                ## prune sequences that do not stabilize
                # xs_cpu, control_values_cpu, indices = prefilter_trajs(xs_cpu, control_values_cpu, min_control_sum=200)
                xs_cpu, control_values_cpu, true_xs_cpu, indices = prefilter_trajs(xs_cpu, control_values_cpu, true_xs_cpu, min_control_sum=200)
                # T_cpu = T_cpu[indices]
                LINK_LENGTH_1_new = LINK_LENGTH_1[indices]
                LINK_LENGTH_2_new = LINK_LENGTH_2[indices]
                LINK_MASS_1_new = LINK_MASS_1[indices]
                LINK_MASS_2_new = LINK_MASS_2[indices]

                ## extract windows around stabilization events
                # xs_cpu, control_values_cpu, batch_idxes, idxes = extract_window_fast(xs_cpu, control_values_cpu, window_size=800)
                # xs_cpu, control_values_cpu, batch_idxes, idxes = extract_window_fast(xs_cpu, control_values_cpu, window_size=600)
                # xs_cpu, control_values_cpu, true_xs_cpu, batch_idxes, idxes = extract_window_fast(xs_cpu, control_values_cpu, true_xs_cpu, window_size=600)
                # # T_cpu = T_cpu[batch_idxes]
                # LINK_LENGTH_1 = LINK_LENGTH_1[batch_idxes]
                # LINK_LENGTH_2 = LINK_LENGTH_2[batch_idxes]
                # LINK_MASS_1 = LINK_MASS_1[batch_idxes]
                # LINK_MASS_2 = LINK_MASS_2[batch_idxes]
                



                save_pickle((xs_cpu, control_values_cpu, LINK_LENGTH_1_new, LINK_LENGTH_2_new, LINK_MASS_1_new, LINK_MASS_2_new, true_xs_cpu), pickle_path)
                append_to_dataset_logger(i-args.training.train_steps, LINK_LENGTH_1_new, LINK_LENGTH_2_new, LINK_MASS_1_new, LINK_MASS_2_new, xs_cpu.shape, test_logger)

                # free gpu memory
                del T, xs, control_values, LINK_LENGTH_1_new, LINK_LENGTH_2_new, LINK_MASS_1_new, LINK_MASS_2_new, control_values_cpu, xs_cpu, true_xs_cpu
                # torch.cuda.empty_cache()
            

        # append_to_seed_file(seed_file, i, seed[0] + i, masses, lengths, k_values, xs.shape)
        curriculum.update()
    
    

def main(args):
    # reseed_all(seed[0])
    global_seed = 42
    random.seed(global_seed)
    make_train_data(args)

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")
    os.makedirs(args.dataset_filesfolder, exist_ok=True)
    with open(os.path.join(args.dataset_filesfolder, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)

