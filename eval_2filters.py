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
    prefix = 'train' if is_train else 'test'
    stats_path = os.path.join(f'{prefix}_statistics_{model}_model/ddim{n_ddim_steps}/', f'{dist}.npz')
    return dict(np.load(stats_path))

def eval_score_with_stats(estimator, id_stats: np.ndarray, ood_stats: np.ndarray):
    score_id = estimator.score_samples(id_stats)
    score_ood = estimator.score_samples(ood_stats)
    return score_id, score_ood

def eval_auroc_with_score(score_id, score_ood):
    y_id = np.ones(score_id.shape[0])
    y_ood = np.zeros(score_ood.shape[0])
    y_true = np.append(y_id, y_ood)
    sample_score = np.append(score_id, score_ood)
    auroc = roc_auc_score(y_true, sample_score)
    return auroc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='celeba', help="Base distribution of model")
    parser.add_argument('--in_dist', type=str, required=True, help='In distribution')
    parser.add_argument('--out_of_dist', type=str, required=True, help='Out of distribution type')
    parser.add_argument('--n_ddim_steps', type=int, default=10, help='Number of ddim steps')
    # Subset parameters
    parser.add_argument('--num_filter2_samples', type=int, default=0, help='The number of pseudo-id samples to add into train set')
    parser.add_argument('--num_train_samples', type=int, default=100, help='The number of train samples to fit GMM')
    parser.add_argument('--num_filter1_samples', type=int, default=100, help='The number of farthest samples to get from filter1')
    args = parser.parse_args()

    redirect_log(args.model, args.n_ddim_steps, args.in_dist, args.out_of_dist,
                 args.num_train_samples, args.num_filter1_samples, args.num_filter2_samples)

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

    print('---------------------------------------------------------------------------------')
    print(f'Calculating scores and AUROC with {args.num_train_samples} training samples')
    naive_6d_estimator = gridsearch_and_fit(gmm_clf, param_grid, train_stats_6d)
    naive_1d_estimator = get_1d_estimator(train_stats_1d)

    id_test_stats_whole = load_whole_stats(args.model, args.n_ddim_steps, args.in_dist, False)
    id_test_stats_6d = load_6d_statistics(id_test_stats_whole)
    id_test_stats_1d = load_1d_statistics(id_test_stats_whole)
    ood_test_stats_whole = load_whole_stats(args.model, args.n_ddim_steps, args.out_of_dist, False)
    ood_test_stats_6d = load_6d_statistics(ood_test_stats_whole)
    ood_test_stats_1d = load_1d_statistics(ood_test_stats_whole)

    # First evaluation with train samples only
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

    print('\n\n')
    print('---------------------------------------------------------------------------------')
    print('Finding outliers with DiffPath-1d estimator')
    # Get the outliers with 1d estimator
    # Separate the outliers to False Negative and True Negative indices
    fn_indices, tn_indices = get_topk_score_indices(naive_1d_id_score, naive_1d_ood_score, args.num_filter1_samples, False)
    print(f'Among {args.num_filter1_samples} outliers, {fn_indices.shape[0]} are actually ID samples')
    outlier_6d_id_stats, outlier_6d_ood_stats = id_test_stats_6d[fn_indices], ood_test_stats_6d[tn_indices]
    outlier_6d_id_score, outlier_6d_ood_score = naive_6d_id_score[fn_indices], naive_6d_ood_score[tn_indices]
    print('Evaluating ID samples with DiffPath-6d estimator from outliers')
    pseudo_id_stats_6d, pseudo_id_score_6d, pseudo_id_label_6d, tp_ratio_6d = get_topk_score_stats(
        outlier_6d_id_stats, outlier_6d_id_score,
        outlier_6d_ood_stats, outlier_6d_ood_score,
        args.num_filter2_samples, True
    )
    print(f'Among {args.num_filter2_samples} pseudo-id samples, {np.sum(pseudo_id_label_6d)} are actually ID samples')

    # Enhance train stats with pseudo-id stats
    enhanced_train_stats_6d = enhance_train_stats(train_stats_6d, pseudo_id_stats_6d, args.num_filter2_samples)
    print(f'Fitting new GMM estimator with enhanced training stats {enhanced_train_stats_6d.shape[0]}')
    enhanced_6d_estimator = gridsearch_and_fit(gmm_clf, param_grid, enhanced_train_stats_6d)
    enhanced_6d_id_score, enhanced_6d_ood_score = eval_score_with_stats(enhanced_6d_estimator, id_test_stats_6d, ood_test_stats_6d)
    print('Enhanced id score evaluated with 6d estimator: ', enhanced_6d_id_score.mean())
    print('Enhanced ood score evaluated with 6d estimator: ', enhanced_6d_ood_score.mean())
    enhanced_6d_auroc = eval_auroc_with_score(enhanced_6d_id_score, enhanced_6d_ood_score)
    print('AUROC after enhancement evaluated with 6d estimator: ', enhanced_6d_auroc)

    outlier_1d_id_stats, outlier_1d_ood_stats = id_test_stats_1d[fn_indices], ood_test_stats_1d[tn_indices]
    outlier_1d_id_score, outlier_1d_ood_score = naive_1d_id_score[fn_indices], naive_1d_ood_score[tn_indices]
    print('==================================================================================================')
    print('Evaluating ID samples with DiffPath-1d estimator from outliers')
    pseudo_id_stats_1d, pseudo_id_score_1d, pseudo_id_label_1d, tp_ratio_1d = get_topk_score_stats(
        outlier_1d_id_stats, outlier_1d_id_score,
        outlier_1d_ood_stats, outlier_1d_ood_score,
        args.num_filter2_samples, True
    )
    print(f'Among {args.num_filter2_samples} pseudo-id samples, {np.sum(pseudo_id_label_1d)} are actually ID samples')
    # Enhance train stats with pseudo-id stats
    enhanced_train_stats_1d = enhance_train_stats(train_stats_1d, pseudo_id_stats_1d, args.num_filter2_samples)
    print(f'Fitting new 1d estimator with enhanced training stats {enhanced_train_stats_1d.shape[0]}')
    enhanced_1d_estimator = get_1d_estimator(enhanced_train_stats_1d)
    enhanced_1d_id_score, enhanced_1d_ood_score = eval_score_with_stats(enhanced_1d_estimator, id_test_stats_1d, ood_test_stats_1d)
    print('Enhanced id score evaluated with 1d estimator: ', enhanced_1d_id_score.mean())
    print('Enhanced ood score evaluated with 1d estimator: ', enhanced_1d_ood_score.mean())
    enhanced_1d_auroc = eval_auroc_with_score(enhanced_1d_id_score, enhanced_1d_ood_score)
    print('AUROC after enhancement evaluated with 1d estimator: ', enhanced_1d_auroc)

    # Use DiffPath-6d as the first filter``
    print('\n\n')
    print('---------------------------------------------------------------------------------')
    print('Finding outliers with DiffPath-6d estimator')
    # Get the outliers with 1d estimator
    # Separate the outliers to False Negative and True Negative indices
    fn_indices, tn_indices = get_topk_score_indices(naive_6d_id_score, naive_6d_ood_score, args.num_filter1_samples, False)
    print(f'Among {args.num_filter1_samples} outliers, {fn_indices.shape[0]} are actually ID samples')
    outlier_6d_id_stats, outlier_6d_ood_stats = id_test_stats_6d[fn_indices], ood_test_stats_6d[tn_indices]
    outlier_6d_id_score, outlier_6d_ood_score = naive_6d_id_score[fn_indices], naive_6d_ood_score[tn_indices]
    print('Evaluating ID samples with DiffPath-6d estimator from outliers')
    pseudo_id_stats_6d, pseudo_id_score_6d, pseudo_id_label_6d, tp_ratio_6d = get_topk_score_stats(
        outlier_6d_id_stats, outlier_6d_id_score,
        outlier_6d_ood_stats, outlier_6d_ood_score,
        args.num_filter2_samples, True
    )
    print(f'Among {args.num_filter2_samples} pseudo-id samples, {np.sum(pseudo_id_label_6d)} are actually ID samples')

    # Enhance train stats with pseudo-id stats
    enhanced_train_stats_6d = enhance_train_stats(train_stats_6d, pseudo_id_stats_6d, args.num_filter2_samples)
    print(f'Fitting new GMM estimator with enhanced training stats {enhanced_train_stats_6d.shape[0]}')
    enhanced_6d_estimator = gridsearch_and_fit(gmm_clf, param_grid, enhanced_train_stats_6d)
    enhanced_6d_id_score, enhanced_6d_ood_score = eval_score_with_stats(enhanced_6d_estimator, id_test_stats_6d, ood_test_stats_6d)
    print('Enhanced id score evaluated with 6d estimator: ', enhanced_6d_id_score.mean())
    print('Enhanced ood score evaluated with 6d estimator: ', enhanced_6d_ood_score.mean())
    enhanced_6d_auroc = eval_auroc_with_score(enhanced_6d_id_score, enhanced_6d_ood_score)
    print('AUROC after enhancement evaluated with 6d estimator: ', enhanced_6d_auroc)

    outlier_1d_id_stats, outlier_1d_ood_stats = id_test_stats_1d[fn_indices], ood_test_stats_1d[tn_indices]
    outlier_1d_id_score, outlier_1d_ood_score = naive_1d_id_score[fn_indices], naive_1d_ood_score[tn_indices]
    print('==================================================================================================')
    print('Evaluating ID samples with DiffPath-1d estimator from outliers')
    pseudo_id_stats_1d, pseudo_id_score_1d, pseudo_id_label_1d, tp_ratio_1d = get_topk_score_stats(
        outlier_1d_id_stats, outlier_1d_id_score,
        outlier_1d_ood_stats, outlier_1d_ood_score,
        args.num_filter2_samples, True
    )
    print(f'Among {args.num_filter2_samples} pseudo-id samples, {np.sum(pseudo_id_label_1d)} are actually ID samples')
    # Enhance train stats with pseudo-id stats
    enhanced_train_stats_1d = enhance_train_stats(train_stats_1d, pseudo_id_stats_1d, args.num_filter2_samples)
    print(f'Fitting new 1d estimator with enhanced training stats {enhanced_train_stats_1d.shape[0]}')
    enhanced_1d_estimator = get_1d_estimator(enhanced_train_stats_1d)
    enhanced_1d_id_score, enhanced_1d_ood_score = eval_score_with_stats(enhanced_1d_estimator, id_test_stats_1d, ood_test_stats_1d)
    print('Enhanced id score evaluated with 1d estimator: ', enhanced_1d_id_score.mean())
    print('Enhanced ood score evaluated with 1d estimator: ', enhanced_1d_ood_score.mean())
    enhanced_1d_auroc = eval_auroc_with_score(enhanced_1d_id_score, enhanced_1d_ood_score)
    print('AUROC after enhancement evaluated with 1d estimator: ', enhanced_1d_auroc)




if __name__ == '__main__':
    main()
