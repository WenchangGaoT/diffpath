'''
  Script for estimating relevance of two dimensions
'''

import warnings 
import argparse
import matplotlib.pyplot as plt

import numpy as np 

from sklearn.exceptions import ConvergenceWarning
from eval_6d import load_6d_statistics, gridsearch_and_fit, eval_auroc_with_stats
from eval_1d import load_1d_statistics, get_1d_estimator
from eval_2filters import load_whole_stats
from subset_utils import *
from math import ceil, floor
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning) 

def plot_all_dims(args):
  id_stats_dict = load_whole_stats(args.model, 10, args.in_dist, True)
  ood_stats_dict = load_whole_stats(args.model, 10, args.out_of_dist, True)
  keys = id_stats_dict.keys()
  all_bins_dict, id_hist_dict, ood_hist_dict = {}, {}, {}
  for i, k in enumerate(keys):
    num_bins = 1000

    all_left_stats = np.concatenate([id_stats_dict[k], ood_stats_dict[k]])

    all_left_bins = np.linspace(all_left_stats.min(), all_left_stats.max(), num_bins + 1)
    all_bins_dict[k] = all_left_bins

    
    id_left_hist, _ = np.histogram(id_stats_dict[k], bins=all_left_bins)
    ood_left_hist, _ = np.histogram(ood_stats_dict[k], bins=all_left_bins)
    id_hist_dict[k] = id_left_hist
    ood_hist_dict[k] = ood_left_hist

  plot_histograms(all_bins_dict, id_hist_dict, ood_hist_dict, keys)


def plot_histograms(all_left_bins: dict, id_left_hist: dict, ood_left_hist: dict, 
                    keys: list[str]):
  '''
  Plot the densities of id and ood stats in given dimensions.
  '''
  fig, axes = plt.subplots(ceil(len(keys) / 5), 5, figsize=(15, 9))

  for i, k in enumerate(keys):
    # plt.subplot(1, i + 1,i + 1)
    axes[floor(i/5), i%5].plot(all_left_bins[k][:-1], id_left_hist[k]/np.sum(id_left_hist[k]), label="ID", color='blue', drawstyle='steps-post')
    axes[floor(i/5), i%5].plot(all_left_bins[k][:-1], ood_left_hist[k]/np.sum(ood_left_hist[k]), label="OOD", color='red', drawstyle='steps-post')
    # axes[i].xlabel(f"{k}")
    # axes[i].ylabel("Density")
    axes[floor(i/5), i%5].set_title(f"Histogram of {k}")
    axes[floor(i/5), i%5].legend()

  plt.tight_layout()
  plt.show()
  
def plot_selected_density(all_left_bins: dict, id_left_hist: dict, ood_left_hist: dict, 
                          id_selected_hist: dict, ood_selected_hist: dict,
                          keys: list[str]):
  '''
  Plot the densities of selected samples within a range in <left_dim> in all other dimensions included in <keys>
  '''
  fig, axes = plt.subplots(ceil(len(keys) / 5), 5, figsize=(15, 9))

  for i, k in enumerate(keys):
    # plt.subplot(1, i + 1,i + 1)
    axes[floor(i/5), i%5].plot(all_left_bins[k][:-1], id_left_hist[k]/np.sum(id_left_hist[k]), label="ID", color='blue', drawstyle='steps-post')
    axes[floor(i/5), i%5].plot(all_left_bins[k][:-1], id_selected_hist[k]/np.sum(id_selected_hist[k]), label="ID Selected samples", color='green', drawstyle='steps-post')
    axes[floor(i/5), i%5].plot(all_left_bins[k][:-1], ood_selected_hist[k]/np.sum(ood_selected_hist[k]), label="OOD Selected samples", color='pink', drawstyle='steps-post')
    axes[floor(i/5), i%5].plot(all_left_bins[k][:-1], ood_left_hist[k]/np.sum(ood_left_hist[k]), label="OOD", color='red', drawstyle='steps-post')
    # axes[i].xlabel(f"{k}")
    # axes[i].ylabel("Density")
    axes[floor(i/5), i%5].set_title(f"Density of {k}")
    axes[floor(i/5), i%5].legend()

  plt.tight_layout()
  plt.show()

def extract_indices_within_range(array: np.ndarray, min_value: float, max_value: float): 
  '''
  Get the indices of samples whose values are within range [<min_value>, <max_value>]
  '''
  return np.where((array >= min_value) & (array <= max_value))[0]


def main(args): 
  # plot_all_dims(args)
  id_stats_dict = load_whole_stats(args.model, 10, args.in_dist, True)
  ood_stats_dict = load_whole_stats(args.model, 10, args.out_of_dist, True)  

  id_left_values = id_stats_dict[args.left_dim]
  id_right_values = id_stats_dict[args.right_dim]
  ood_right_values = ood_stats_dict[args.right_dim]
  ood_left_values = ood_stats_dict[args.left_dim]

  id_indices = extract_indices_within_range(id_left_values, args.min_value, args.max_value)
  ood_indices = extract_indices_within_range(ood_left_values, args.min_value, args.max_value)
  print(f'{len(id_indices)} out of {len(id_stats_dict[args.left_dim])}samples lie in range [{args.min_value}, {args.max_value}] in dimension {args.left_dim}')
  print(f'{len(ood_indices)} out of {len(ood_stats_dict[args.left_dim])}samples lie in range [{args.min_value}, {args.max_value}] in dimension {args.left_dim}')
  np.random.shuffle(id_indices)
  np.random.shuffle(ood_indices)
  id_indices = id_indices[-10:]
  ood_indices = ood_indices[-10:]
  # indices = extract_indices_within_range(ood_left_values, args.min_value, args.max_value)

  # num_bins = 1000
  # all_left_stats = np.concatenate([id_left_values, ood_left_values])
  # all_left_bins = np.linspace(all_left_stats.min(), all_left_stats.max(), num_bins+1)
  # all_right_stats = np.concatenate([id_right_values, ood_right_values])
  # all_right_bins = np.linspace(all_right_stats.min(), all_right_stats.max(), num_bins+1)
  
  # id_left_hist = np.histogram(id_left_values, all_left_bins)
  # ood_left_hist = np.histogram(ood_left_hist, all_left_bins)

  # id_right_hist = np.histogram(id_right_values, all_right_bins)
  # ood_right_hist = np.histogram(ood_right_hist, all_right_bins)
  keys = id_stats_dict.keys()
  all_bins_dict, id_hist_dict, ood_hist_dict = {}, {}, {}
  id_selected_hist_dict = {}
  ood_selected_hist_dict = {}
  for i, k in enumerate(keys):
    num_bins = 1000

    all_left_stats = np.concatenate([id_stats_dict[k], ood_stats_dict[k]])

    all_left_bins = np.linspace(all_left_stats.min(), all_left_stats.max(), num_bins + 1)
    all_bins_dict[k] = all_left_bins

    id_selected_stats = id_stats_dict[k][id_indices]
    ood_selected_stats = ood_stats_dict[k][ood_indices]
    # selected_stats = ood_stats_dict[k][indices]
    id_left_hist, _ = np.histogram(id_stats_dict[k], bins=all_left_bins)
    ood_left_hist, _ = np.histogram(ood_stats_dict[k], bins=all_left_bins)
    id_selected_hist, _ = np.histogram(id_selected_stats, bins=all_left_bins)
    ood_selected_hist, _ = np.histogram(ood_selected_stats, bins=all_left_bins)
    id_hist_dict[k] = id_left_hist
    ood_hist_dict[k] = ood_left_hist
    id_selected_hist_dict[k] = id_selected_hist
    ood_selected_hist_dict[k] = ood_selected_hist

  plot_selected_density(all_bins_dict, id_hist_dict, ood_hist_dict, id_selected_hist_dict, ood_selected_hist_dict, keys)


if __name__ == '__main__':
  parser = argparse.ArgumentParser() 
  parser.add_argument('--in_dist', type=str, default='cifar10', help='ID dataset name')
  parser.add_argument('--out_of_dist', type=str, default='svhn', help='OOD dataset name')
  parser.add_argument('--left_dim', type=str, default='eps_sum_sq', help='Left dimension name for evaluation')
  parser.add_argument('--right_dim', type=str, default='deps_dt_sq_sqrt', help='Right dimension name for evaluation')
  parser.add_argument('--min_value', type=float, default=100, help='Select a range for left dim')
  parser.add_argument('--max_value', type=float, default=200, help='Select a range for left dim')
  parser.add_argument('--model', type=str, default='celeba', help='Celeba or Imagenet')
  args = parser.parse_args()
  plot_all_dims(args)
  main(args)
