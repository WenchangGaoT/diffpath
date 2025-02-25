import sys
import argparse
import os

import numpy as np

from eval_6d import load_6d_statistics, gridsearch_and_fit, eval_auroc_with_stats
from eval_1d import load_1d_statistics, get_1d_estimator
from subset_utils import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score


def redirect_log(model, n_ddim_steps, in_dist, out_dist,
                 num_train_samples, num_filter1_samples, num_filter2_samples):
    '''
    Redirects the outputs
    '''
    log_dir = f'logs/{model}_model/ddim{n_ddim_steps}/{in_dist}_vs_{out_dist}/'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir,
                            f'train_{num_train_samples}-filter1_{num_filter1_samples}-filter2_{num_filter2_samples}.txt')
    print(f'Outputs are redirected to {log_path}')
    sys.stdout = open(log_path, 'w+')

def load_whole_stats(model, n_ddim_steps, dist, is_train=True):
    '''
    Load the whole 7-D dataset from `${train}_statistics_${model}_model/ddim${n_ddim_steps}/${dist}`.

    Return the loaded stats as a dictionary
    '''
    prefix = 'train' if is_train else 'test'
    stats_path = os.path.join(f'{prefix}_statistics_{model}_model/ddim{n_ddim_steps}/', f'{dist}.npz')
    return dict(np.load(stats_path))

def eval_score_with_stats(estimator, id_stats: np.ndarray, ood_stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
    '''
    Evaluate the score of ID dataset and OOD dataset according to their stats with a given estimator.

    Params: 
        estimator: sklearn.
    '''
    score_id = estimator.score_samples(id_stats)
    score_ood = estimator.score_samples(ood_stats)
    return score_id, score_ood

def eval_auroc_with_score(score_id: np.ndarray, score_ood: np.ndarray):
    '''
    Evaluate the auroc given ID set score and OOD set score.

    Params: 
        <score_id>: np.ndarray of shape (N1,), the score array of ID set 
        <score_ood>: np.ndarray of shape (N2,), the score array of OOD set

    Return: 
        A float, AUROC of current model
    '''
    y_id = np.ones(score_id.shape[0])
    y_ood = np.zeros(score_ood.shape[0])
    y_true = np.append(y_id, y_ood)
    sample_score = np.append(score_id, score_ood)
    auroc = roc_auc_score(y_true, sample_score)
    return auroc

def get_pseudo_id_stats(id_score_first: np.ndarray, 
                        ood_score_first: np.ndarray, 
                        id_score_second: np.ndarray, 
                        ood_score_second: np.ndarray, 
                        id_stats_dict: dict, 
                        ood_stats_dict: dict,
                        n_outlier_samples: int, 
                        n_pid_samples: int) -> tuple: 
    '''
    Given the scores for outlier detection (first) and for pseudo-id detection (second), 
    return the p-id sample stats. 

    Params: 
        <id_score_first>: (N1, D) the score array of ID set for outlier detection 
        <ood_score_first>: (N2, D) the score array of OOD set for outlier detection 
        <id_score_second>: the score array of ID set for p-id selection 
        <ood_score_second>: the score array of OOD set for p-id selection 
        <id_stats_dict>: a dictionary of keys ("1d", "6d") storing ID test stats 
        <ood_stats_dict>: a dictionary of keys ("1d", "6d") storing OOD test stats
        <n_outlier_samples>: int, the number of outlier samples wanted
        <n_pid_samples: int, the number of p-id samples wanted
    Return:
        A dictionary {
            "1d": np.ndarray of shape (<n_pid_samples>, 1)
            "6d": np.ndarray of shape (<n_pid_samples>, 6)
        }

    '''
    # Get outliers from the first scores
    naive_id_stats_1d, naive_id_stats_6d = id_stats_dict["1d"], id_stats_dict["6d"]
    naive_ood_stats_1d, naive_ood_stats_6d = ood_stats_dict["1d"], ood_stats_dict["6d"]
    fn_indices, tn_indices = get_topk_score_indices(id_score_first, 
                                                    ood_score_first, 
                                                    n_outlier_samples, 
                                                    False)
    (id_score_second,
    id_outlier_stats_1d,
    id_outlier_stats_6d) = id_score_second[fn_indices], naive_id_stats_1d[fn_indices], naive_id_stats_6d[fn_indices]
    (ood_score_second, 
     ood_outlier_stats_1d, 
     ood_outlier_stats_6d) = ood_score_second[tn_indices], naive_ood_stats_1d[fn_indices], naive_ood_stats_6d[fn_indices]
    pid_tp_indices, pid_fp_indices = get_topk_score_indices(id_score_second, 
                                                            ood_score_second, 
                                                            n_pid_samples, 
                                                            True) 
    print(f'Among {n_pid_samples} pseudo-ID samples, {pid_tp_indices.shape[0]} are ID samples ')
    pseudo_id_stats_1d = merge_stats_with_indices(id_outlier_stats_1d, pid_tp_indices, 
                                                  ood_outlier_stats_1d, pid_fp_indices)
    pseudo_id_stats_6d = merge_stats_with_indices(id_outlier_stats_6d, pid_tp_indices, 
                                                  ood_outlier_stats_6d, pid_fp_indices)
    return {
        "1d": pseudo_id_stats_1d, 
        "6d": pseudo_id_stats_6d
    }

def main():
    # Load in train stats then select a subset randomly
    # Splits the 7-D stats into 6-D to fit DiffPath6D classifier and 6-D for DiffPath1D classifier
    train_stats_whole = load_whole_stats(args.model, args.n_ddim_steps, args.in_dist, True)
    train_stats_whole = get_subset_stats(train_stats_whole, args.num_train_samples)
    print(f'Train stats: {train_stats_whole["deps_dt_sq_sqrt"].shape}')
    train_stats_1d = load_1d_statistics(train_stats_whole)
    train_stats_6d = load_6d_statistics(train_stats_whole)

    # Params GMM grid search for 6D classifier
    param_grid = {
        'GMM__n_components': [3, 5, 10, 20, 50, 100],
        'GMM__covariance_type': ['full', 'tied', 'diag', 'spherical'],
        'GMM__max_iter': [100, 200, 300, 500]
    }
    gmm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('GMM', GaussianMixture())
    ])

    # Fit two classifiers with subset train samples
    print('-' * 80)
    print(f'Calculating scores and AUROC with {args.num_train_samples} training samples')
    assert train_stats_6d.shape[0] == args.num_train_samples 
    assert train_stats_1d.shape[0] == args.num_train_samples
    naive_6d_estimator = gridsearch_and_fit(gmm_clf, param_grid, train_stats_6d)
    naive_1d_estimator = get_1d_estimator(train_stats_1d)

    # Load test ID and OOD stats of test sets
    id_test_stats_whole = load_whole_stats(args.model, args.n_ddim_steps, args.in_dist, False)
    id_test_stats_6d = load_6d_statistics(id_test_stats_whole)
    id_test_stats_1d = load_1d_statistics(id_test_stats_whole)
    ood_test_stats_whole = load_whole_stats(args.model, args.n_ddim_steps, args.out_of_dist, False)
    ood_test_stats_6d = load_6d_statistics(ood_test_stats_whole)
    ood_test_stats_1d = load_1d_statistics(ood_test_stats_whole)

    id_test_stats_dict = {
        '1d': id_test_stats_1d, 
        '6d': id_test_stats_6d
    }
    ood_test_stats_dict = {
        '1d': ood_test_stats_1d, 
        '6d': ood_test_stats_6d
    }

    auroc_dict = {}

    # Chapter 1: First evaluation: 
    # Estimate test sample scores with two fitted classifiers
    # Note that the classifiers tend to be overfitted as number of samples is limited
    naive_1d_id_score, naive_1d_ood_score = eval_score_with_stats(naive_1d_estimator, id_test_stats_1d, ood_test_stats_1d)
    naive_1d_auroc = eval_auroc_with_score(naive_1d_id_score, naive_1d_ood_score)
    naive_6d_id_score, naive_6d_ood_score = eval_score_with_stats(naive_6d_estimator, id_test_stats_6d, ood_test_stats_6d)
    naive_6d_auroc = eval_auroc_with_score(naive_6d_id_score, naive_6d_ood_score)
    print('ID score with DiffPath-1d statistics: ', naive_1d_id_score.mean())
    print('OOD score with DiffPath-1d statistics: ', naive_1d_ood_score.mean())
    print('ID score with DiffPath-6d statistics: ', naive_6d_id_score.mean())
    print('OOD score with DiffPath-6d statistics: ', naive_6d_ood_score.mean())
    print('AUROC with DiffPath-1d statistics: ', naive_1d_auroc)
    print('AUROC with DiffPath-6d statistics: ', naive_6d_auroc)

    auroc_dict['naive_1d'] = naive_1d_auroc 
    auroc_dict['naive_6d'] = naive_6d_auroc
    # First evaluation finishes here

    # Chapter 2: Pseudo-ID set enhancement: 
    # Since the classifiers tend to be overfitted, the outliers from the first evaluation
    # contain both OOD and ID samples. These misclassified ID samples are more informative. 
    # In this stage, we augment the train set by introducing pseufo-ID samples from the 
    # outliers. We use one classifier (DP-6D) to find outliers, and another classifier (DP-1D)
    # to find pseudo-ID (False Negative) samples and re-fit classifiers with enhanced dataset

    ############################################################################
    # Experiment 1: 
    # 1D outlier detection -> 6D p-id detection
    # Find outliers with DiffPath-1D Estimator
    print('\n\n')
    print('=' * 80)
    print('Finding outliers with DiffPath-1d estimator\nFinding pseudo-ID samples with DiffPath-6d')
    # # Experiment 1-1: 
    # # 1D outlier detection -> 6D p-id detection -> 1D final classification
    pid_stats_dict = get_pseudo_id_stats(naive_1d_id_score,
                                         naive_1d_ood_score, 
                                         naive_6d_id_score, 
                                         naive_6d_ood_score, 
                                         id_test_stats_dict, 
                                         ood_test_stats_dict, 
                                         args.num_filter1_samples, 
                                         args.num_filter2_samples)

    enhanced_train_stats_1d = enhance_train_stats(train_stats_1d, 
                                                  pid_stats_dict['1d'], 
                                                  args.num_filter2_samples)
    enhanced_1d_estimator = get_1d_estimator(enhanced_train_stats_1d)
    enhanced_1d_id_score, enhanced_1d_ood_score = eval_score_with_stats(enhanced_1d_estimator, 
                                                                        id_test_stats_1d, 
                                                                        ood_test_stats_1d)
    print('Enhanced id score evaluated with 1d estimator: ', enhanced_1d_id_score.mean())
    print('Enhanced ood score evaluated with 1d estimator: ', enhanced_1d_ood_score.mean())
    enhanced_1d_auroc = eval_auroc_with_score(enhanced_1d_id_score, enhanced_1d_ood_score)
    print('AUROC after enhancement evaluated with 1d estimator: ', enhanced_1d_auroc)
    print('-' * 80)
    # Experiment 1-1 finishes here
    # Experiment 1-2: 
    # 1D outlier detection -> 6D p-id detection -> 6D final classification
    # Enhance pseudo-id set with p-id samples
    enhanced_train_stats_6d = enhance_train_stats(train_stats_6d, 
                                                  pid_stats_dict['6d'], 
                                                  args.num_filter2_samples)
    enhanced_6d_estimator = gridsearch_and_fit(gmm_clf, param_grid, enhanced_train_stats_6d)
    enhanced_6d_id_score, enhanced_6d_ood_score = eval_score_with_stats(enhanced_6d_estimator, 
                                                                        id_test_stats_6d, 
                                                                        ood_test_stats_6d)
    print('Enhanced id score evaluated with 6d estimator: ', enhanced_6d_id_score.mean())
    print('Enhanced ood score evaluated with 6d estimator: ', enhanced_6d_ood_score.mean())
    enhanced_6d_auroc = eval_auroc_with_score(enhanced_6d_id_score, enhanced_6d_ood_score)
    print('AUROC after enhancement evaluated with 6d estimator: ', enhanced_6d_auroc)
    auroc_dict['1d->6d->1d'] = enhanced_1d_auroc 
    auroc_dict['1d->6d->6d'] = enhanced_6d_auroc
    # Experiment 1-2 ends here
    # Experiment 1 ends here
    ###########################################################################

    ###########################################################################
    # Experiment 2: 
    # 6D outlier detection -> 1D p-id detectiona
    print('\n\n')
    print('=' * 80)
    print('Finding outliers with DiffPath-6d estimator')
    print('Finding pseudo-ID samples with DiffPath-1d')
    # Experiment 2-1: 
    # 6D outlier detection -> 1D p-id detection -> 1D final classification
    pid_stats_dict = get_pseudo_id_stats(naive_6d_id_score,
                                         naive_6d_ood_score, 
                                         naive_1d_id_score, 
                                         naive_1d_ood_score, 
                                         id_test_stats_dict, 
                                         ood_test_stats_dict, 
                                         args.num_filter1_samples, 
                                         args.num_filter2_samples)
    enhanced_train_stats_1d = enhance_train_stats(train_stats_1d, 
                                                  pid_stats_dict['1d'], 
                                                  args.num_filter2_samples)
    enhanced_1d_estimator = get_1d_estimator(enhanced_train_stats_1d)
    enhanced_1d_id_score, enhanced_1d_ood_score = eval_score_with_stats(enhanced_1d_estimator, 
                                                                        id_test_stats_1d, 
                                                                        ood_test_stats_1d)
    print('Enhanced id score evaluated with 1d estimator: ', enhanced_1d_id_score.mean())
    print('Enhanced ood score evaluated with 1d estimator: ', enhanced_1d_ood_score.mean())
    enhanced_1d_auroc = eval_auroc_with_score(enhanced_1d_id_score, enhanced_1d_ood_score)
    print('AUROC after enhancement evaluated with 1d estimator: ', enhanced_1d_auroc)
    print('-' * 80)
    # Experiment 2-1 finishes here
    # Experiment 2-2: 
    # 6D outlier detection -> 1D p-id detection -> 6D final classification
    # Enhance pseudo-id set with p-id samples
    enhanced_train_stats_6d = enhance_train_stats(train_stats_6d, 
                                                  pid_stats_dict['6d'], 
                                                  args.num_filter2_samples)
    enhanced_6d_estimator = gridsearch_and_fit(gmm_clf, param_grid, enhanced_train_stats_6d)
    enhanced_6d_id_score, enhanced_6d_ood_score = eval_score_with_stats(enhanced_6d_estimator, 
                                                                        id_test_stats_6d, 
                                                                        ood_test_stats_6d)
    print('Enhanced id score evaluated with 6d estimator: ', enhanced_6d_id_score.mean())
    print('Enhanced ood score evaluated with 6d estimator: ', enhanced_6d_ood_score.mean())
    enhanced_6d_auroc = eval_auroc_with_score(enhanced_6d_id_score, enhanced_6d_ood_score)
    print('AUROC after enhancement evaluated with 6d estimator: ', enhanced_6d_auroc)

    auroc_dict['6d->1d->1d'] = enhanced_1d_auroc 
    auroc_dict['6d->1d->6d'] = enhanced_6d_auroc
    return auroc_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='celeba', help="Base distribution of model")
    parser.add_argument('--in_dist', type=str, required=True, help='In distribution')
    parser.add_argument('--out_of_dist', type=str, required=True, help='Out of distribution type')
    parser.add_argument('--n_ddim_steps', type=int, default=10, help='Number of ddim steps')
    # Subset parameters
    parser.add_argument('--num_train_samples', type=int, default=100, 
                        help='The number of train samples to fit GMM')
    parser.add_argument('--num_filter1_samples', type=int, default=100, 
                        help='The number of outliers samples to get from filter1')
    parser.add_argument('--num_filter2_samples', type=int, default=10, 
                        help='The number of pseudo-id samples to add into train set')
    
    parser.add_argument('--num_experiment_trials', type=int, default=1, 
                        help='The number of experiments to run')
    args = parser.parse_args()

    redirect_log(args.model, args.n_ddim_steps, args.in_dist, args.out_of_dist,
                 args.num_train_samples, args.num_filter1_samples, args.num_filter2_samples)
    
    auroc_naive_1d_ls, auroc_naive_6d_ls = [], [] 
    auroc_161_ls, auroc_166_ls = [], [] 
    auroc_611_ls, auroc_616_ls = [], []
    
    for _ in range(args.num_experiment_trials): 
        auroc_dict = main()
        auroc_naive_1d_ls.append(auroc_dict['naive_1d'])
        auroc_naive_6d_ls.append(auroc_dict['naive_6d'])
        auroc_161_ls.append(auroc_dict['1d->6d->1d'])
        auroc_166_ls.append(auroc_dict['1d->6d->6d'])
        auroc_611_ls.append(auroc_dict['6d->1d->1d'])
        auroc_616_ls.append(auroc_dict['6d->1d->6d'])
    
    print('=' * 80)
    print(f'Final AUROC results in {args.num_experiment_trials} trials: (mean\tvariance)')
    print(f'Naive DiffPath-1d:\t{np.mean(auroc_naive_1d_ls)}\t{np.var(auroc_naive_1d_ls)}')
    print(f'Naive DiffPath-6d:\t{np.mean(auroc_naive_6d_ls)}\t{np.var(auroc_naive_6d_ls)}')
    print(f'1->6->1 DP-1d: \t{np.mean(auroc_161_ls)}\t{np.var(auroc_161_ls)}')
    print(f'1->6->6 DP-1d: \t{np.mean(auroc_166_ls)}\t{np.var(auroc_166_ls)}')
    print(f'6->1->1 DP-1d: \t{np.mean(auroc_611_ls)}\t{np.var(auroc_611_ls)}')
    print(f'6->1->6 DP-1d: \t{np.mean(auroc_616_ls)}\t{np.var(auroc_616_ls)}')
