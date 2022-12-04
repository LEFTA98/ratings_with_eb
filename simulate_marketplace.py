"""
script for running on SOAL cluster - running simulated marketplace for online marketplaces project with different hyperparameter
settings.
"""

import numpy as np
import pandas as pd
import scipy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import argparse
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from scipy.stats import beta

PRIOR_A = 14.270965891341989
PRIOR_B = 5.916156493565227

class ProductHelper:
    """Class for helping keep track of all the products we have."""
    
    def __init__(self, universe, starting_market, mkt_ids, priors):
        self.universe = universe
        self.mkt_ids = mkt_ids
        self.priors = priors
        self.market = [np.array([copy.deepcopy(self.priors)[k]]) for k in starting_market]
        
    def pull_arm(self, action, like):
        arr = self.market[action]
        latest_action = copy.deepcopy(arr[-1])
        latest_action += np.array([like, 1-like])
        self.market[action] = np.append(arr, [latest_action], axis=0)
        
    def pull_arm_update_market(self, action, like):
        self.pull_arm(action, like)
        for i in range(len(self.market)):
            if not np.array_equal(i, action):
                self.market[i] = np.append(self.market[i], [self.market[i][-1]], axis=0)
        
    def replace_item(self, old_id, new_id):
        old_idx = self.mkt_ids.index(old_id)
        self.universe[old_id].append(copy.deepcopy(self.market[old_idx]))
        self.mkt_ids[old_idx] = new_id
        self.market[old_idx] = np.array([copy.deepcopy(self.priors[new_id])])

def random_argmax(alist):
    maxval = max(alist)
    argmax = [idx for idx, val in enumerate(alist) if val == maxval]
    return np.random.choice(argmax)

def ts_action(actions, num_success, num_failure):
    p_hat = [np.random.beta(num_success[a],num_failure[a]) for a in actions]
    a = random_argmax(p_hat)
    return a

def sample_chosen_df(videos, chosen_df, action_index):
    vid = videos[action_index]
    seen_like = chosen_df[chosen_df['video_id']==vid].sample(1).iloc[0]['liked']
    return seen_like

def run_multiarmed_bandit_replenishment(chosen_df,
                                        videos,
                                        priors,
                                        sampling_action,
                                        timesteps,
                                        rho,
                                        mkt_size,
                                        num_users=1,
                                        snapshot_start=None,
                                        snapshotting_prob=0.001,
                                        id_name=None):
    print(f'simulation {id_name} beginning...')
    if not snapshot_start:
        snapshot_start = timesteps//5
    product_data = dict(zip(videos, [[] for _ in range(len(videos))]))
    priors_dict = dict(zip(videos, [priors.copy()[i,:] for i in range(priors.shape[0])]))
    snapshot_dict = dict()
    snapshot_num = 1
            
    curr_vids = np.array(list(np.random.choice(videos, mkt_size, replace=False)))
    remaining_vids = set(videos).difference(set(curr_vids))

    helper = ProductHelper(product_data, curr_vids, list(curr_vids), priors_dict)

    for t in range(timesteps):
        if (t+1) % (timesteps//10) == 0:
            print(f'{t+1}/{timesteps}')
            
        latest_sims = np.array([item[-1] for item in helper.market])
        successes, failures = latest_sims[:,0], latest_sims[:,1]
        actions = range(mkt_size)
        for m in range(num_users):
            a = sampling_action(actions, successes, failures)
            chosen_action_global_index = videos.index(helper.mkt_ids[a])
            like = sample_chosen_df(videos, chosen_df, chosen_action_global_index)

            # update prior
            helper.pull_arm_update_market(a, like)

        # replenish the indices
        flips = np.random.binomial(1, rho, mkt_size)
        draws = np.random.choice(list(remaining_vids), mkt_size, replace=False)

        replenishments = flips * draws
        replaced = flips * helper.mkt_ids
        swapped_pairs = zip(list(replaced[replaced != 0].flatten()), list(replenishments[replenishments != 0].flatten()))
        replenishments = set(replenishments[replenishments != 0].flatten().astype(int))
        replaced = set(replaced[replaced != 0].flatten().astype(int))
        remaining_vids = remaining_vids.union(replaced).difference(replenishments)
        
        # ensure that the remaining videos are distinct size is constant
        if len(list(remaining_vids)) != len(videos) - mkt_size:
            print('remaining_vids', len(list(remaining_vids)))
            print('curr_vids', helper.mkt_ids)
            print('replenishments', replenishments)
            print('replaced', replaced)
            print('flips', flips)
            print('draws', draws)
            assert False

        for old,new in swapped_pairs:
            helper.replace_item(old, new)
            
        if t >= snapshot_start and np.random.binomial(1, snapshotting_prob) > 0:
            snapshot_dict[snapshot_num] = (copy.deepcopy(helper.mkt_ids), copy.deepcopy(helper.market))
            snapshot_num += 1
            
    for prod in helper.mkt_ids:
        mkt_idx = helper.mkt_ids.index(prod)
        helper.universe[prod].append(helper.market[mkt_idx])
    
    #rename everything by id name if given
    if id_name:
        for k in list(helper.universe.keys())[:]:
            helper.universe[(k, id_name)] = helper.universe.pop(k)
            
        for k in list(snapshot_dict.keys())[:]:
            snapshot_dict[(k, id_name)] = snapshot_dict.pop(k)
            
    return helper.universe, snapshot_dict

def setup_data(num_samples=100):
    # setup and transform the data
    kuairec_test = pd.read_csv('kuairec_test.csv')
    
    df = kuairec_test[['video_id', 'like_ratio']].drop_duplicates().sort_values(by='like_ratio', ascending=False)
    test_videos = pd.Series(data=df['like_ratio'])
    test_videos.index = df['video_id']
    
    sampled_videos = (list(test_videos.sample(num_samples, random_state=1729).keys()))
    kuairec_chosen = kuairec_test[kuairec_test['video_id'].isin(sampled_videos)]
    
    return sampled_videos, kuairec_chosen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate simulation outputs for online marketplace experiment.')
    parser.add_argument('--mode', choices=['test', 'full'], default='full')
    parser.add_argument('--timestep', type=int, default=1000000)
    parser.add_argument('--exit_rate', type=float, default=0.01)
    parser.add_argument('--market_size', type=int, default=10)
    parser.add_argument('--universe_size', type=int, default=100)
    args = parser.parse_args()
    
    sampled_videos, kuairec_chosen = setup_data(args.universe_size)
    
    uninformed_priors = np.ones(len(sampled_videos)*2).reshape(len(sampled_videos),2)
    eb_priors = np.array([[PRIOR_A]*len(sampled_videos),[PRIOR_B]*len(sampled_videos)]).T
    
    prior_settings = []
    prior_names = []
    for a in np.linspace(0,1,11):
        curr_prior = (1-a)*uninformed_priors + a*eb_priors
        prior_settings.append(curr_prior)
        prior_names.append(np.round(a, 2))
        
    if args.mode=='test':
        data, snapshots = run_multiarmed_bandit_replenishment(kuairec_chosen,
                                                              sampled_videos,
                                                              prior_settings[0],
                                                              ts_action,
                                                              timesteps=args.timestep,
                                                              rho=args.exit_rate,
                                                              mkt_size=args.market_size,
                                                              id_name = prior_names[0])
        np.save(f'sims/sim_data_alpha_{prior_names[0]}.npy', data)
        np.save(f'sims/sim_snapshots_alpha_{prior_names[0]}.npy', snapshots)
    else:
        parallel = Parallel(n_jobs=11, verbose=10)
        result_data = parallel(delayed(run_multiarmed_bandit_replenishment)(kuairec_chosen,
                                                              sampled_videos,
                                                              prior_settings[i],
                                                              ts_action,
                                                              timesteps=args.timestep,
                                                              rho=args.exit_rate,
                                                              mkt_size=args.market_size,
                                                              id_name = prior_names[i]) \
                                                              for i in range(len(prior_names)))
        
        result_data = list(result_data)
        print(len(result_data))
        for i in range(len(result_data)):
            data, snapshots = result_data[i]
            np.save(f'sims/sim_data_alpha_{prior_names[i]}.npy', data)
            np.save(f'sims/sim_snapshots_alpha_{prior_names[i]}.npy', snapshots)            
    
        