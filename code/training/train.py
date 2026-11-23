import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import subprocess
import sys

import config

max_iters = 1000 # Total number of training episodes
episode_iters = 20 # How many training episodes to execute before the training script is called again

# NOTICE: The training script trains and stores the models every 10 episode_iters. When the batch of episode_iters
# finishes, the model is stored and in the next itreation is loaded again to start the training were it was left before.
# This is to avoid some memory leak issue existing with TF

'''
Master training script of the DRL agent: it trains for a total of 'max_iters' episodes, launching
training.train_worker in batches of 'episode_iters' episodes (to avoid the memory growth of TF).

Usage (from code/):
  python -m training.train
'''


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Master Training Script')
    parser.add_argument('-lr', type=float, default=0.0005)
    parser.add_argument('-T', type=int, default=8)
    parser.add_argument('-dim', type=int, default=32)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-readout_layers', type=int, default=4)
    parser.add_argument('-gamma', type=float, default=0.95)
    parser.add_argument('-vf_coef', type=float, default=0.5)
    parser.add_argument('-start_iter', type=int, default=0, help='Episode from which to resume the training (0 by default)')
    parser.add_argument('-start_c', type=int, default=1, help='Checkpoint counter to start from (1 by default)')
    parser.add_argument('-episode_iters', type=int, default=20, help='Episodes per subprocess')
    parser.add_argument('-max_iters', type=int, default=1000, help='Maximum number of episodes after which the training stops')
    args = parser.parse_args()

    max_iters = args.max_iters
    episode_iters = args.episode_iters

    if not os.path.exists(config.LOGS_DIR):
        os.makedirs(config.LOGS_DIR)

    
    # Temporary folder created when this script starts, just before training the model.
    if not os.path.exists(config.CODE_DIR + "/tmp"):
        os.makedirs(config.CODE_DIR + "/tmp")

    iters = args.start_iter
    counter_store_model = args.start_c
    # Names of the training topologies (do NOT add NEW_ here, the worker script does it)
    dataset_folder_name1 = "BtAsiaPac"
    dataset_folder_name2 = "Garr199905"
    dataset_folder_name3 = "Goodnet"

    while iters < max_iters:
        '''
        Arguments passed to the worker:
            * iters --> current training episode
            * counter_store_model --> new version of the model that will be obtained
            * episode_iters ---> batch of episodes (how many episodes the worker training.train_worker processes)
        '''
        
        cmd = f"\"{sys.executable}\" -m training.train_worker -i {iters} -c {counter_store_model} -e {episode_iters} -f1 {dataset_folder_name1} -f2 {dataset_folder_name2} -f3 {dataset_folder_name3} -lr {args.lr} -T {args.T} -dim {args.dim} -dropout {args.dropout} -readout_layers {args.readout_layers} -gamma {args.gamma} -vf_coef {args.vf_coef}"
        
        try:
            subprocess.check_call(cmd, shell=True, cwd=config.CODE_DIR)
        except subprocess.CalledProcessError as e:
            print(f"\n[FATAL ERROR] The subprocess failed while processing episode {iters}.")
            print(f"The child process finished with error code: {e.returncode}")
            print("Aborting the master DRL training loop to avoid empty iterations and wrongly overwriting models.")
            break
        
        counter_store_model = counter_store_model + episode_iters
        iters = iters + episode_iters






