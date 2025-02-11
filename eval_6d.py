import sys
import numpy as np
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
import os

from subset_utils import *

def load_statistics(d):
    eps = d["eps_sum"]
    eps_sq = d["eps_sum_sq"]
    eps_cb = d["eps_sum_cb"]
    deps_dt = d["deps_dt"]
    deps_dt_sq = d["deps_dt_sq"]
    deps_dt_cb = d["deps_dt_cb"]
    stacked = np.column_stack([eps, eps_sq, eps_cb, deps_dt, deps_dt_sq, deps_dt_cb])
    return stacked

def gridsearch_and_fit(gmm_clf: Pipeline, param_grid: dict, train_stats):
    grid = GridSearchCV(estimator=gmm_clf, param_grid=param_grid, cv=10, n_jobs=5,verbose=1)
    grid_result = grid.fit(train_stats)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    train_score = grid_result.best_score_
    best_gmm = gmm_clf.set_params(**grid.best_params_)
    best_gmm.fit(train_stats)
    return best_gmm

def eval_auroc_with_stats(best_gmm, id_test_stats, ood_test_stats):
    score_test_id = best_gmm.score_samples(id_test_stats)
    score_test_ood = best_gmm.score_samples(ood_test_stats)
    print('id test score: ', score_test_id.mean())
    print('ood test score: ', score_test_ood.mean())

    y_test_id = np.ones(score_test_id.shape[0])
    y_test_ood = np.zeros(score_test_ood.shape[0])
    y_true = np.append(y_test_id, y_test_ood)

    sample_score = np.append(score_test_id, score_test_ood)
    auroc = roc_auc_score(y_true, sample_score)

    return auroc



def main():

    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument('--model', type=str, default='celeba', help="Base distribution of model")
    parser.add_argument('--in_dist', type=str, required=True, help='In distribution')
    parser.add_argument('--out_of_dist', type=str, required=True, help='Out of distribution type')
    parser.add_argument('--n_ddim_steps', type=int, default=10, help='Number of ddim steps')
    # Subset parameters
    parser.add_argument('--enhance_ratio', type=float, default=0, help='The ratio of pseudo-id samples to add into train set')
    parser.add_argument('--subset_ratio', type=float, default=1.0, help='The ratio of subset to whole train set')
    args = parser.parse_args()

    # Redirect the outputs
    log_dir = f'logs/{args.model}_model/ddim{args.n_ddim_steps}'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir,
                            f'{args.in_dist}_{args.subset_ratio}_vs_{args.out_of_dist}.txt')
    sys.stdout = open(log_path, 'w+')
    # load in distribution training dataset statistics
    # Added subset ratio parameter into the path
    in_dist_train_path = os.path.join(f"train_statistics_{args.model}_model/ddim{args.n_ddim_steps}/subset_{args.subset_ratio}",
                                      args.in_dist) + ".npz"
    print(f"Loading in dist train statistics from {in_dist_train_path}")
    in_dist_train_statistics_file = np.load(in_dist_train_path)
    id_train_statistics = load_statistics(in_dist_train_statistics_file) # (N, 6)

    # grid search for best GMM params
    param_grid = {
        'GMM__n_components': [5, 10, 20, 50, 100],  # different numbers of clusters
        'GMM__covariance_type': ['full', 'tied', 'diag', 'spherical'],  # different types of covariance
        'GMM__max_iter': [200]
    }
    gmm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("GMM", GaussianMixture())
    ])

    best_gmm = gridsearch_and_fit(gmm_clf, param_grid, id_train_statistics)
#     grid = GridSearchCV(estimator=gmm_clf, param_grid=param_grid, cv=10, n_jobs=5,verbose=1)
#     grid_result = grid.fit(id_train_statistics)
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     train_score = grid_result.best_score_
#     best_gmm = gmm_clf.set_params(**grid.best_params_)
#     best_gmm.fit(id_train_statistics)

    # load in distribution test dataset statistics
    in_dist_test_path = os.path.join(f"test_statistics_{args.model}_model/ddim{args.n_ddim_steps}/", args.in_dist) + ".npz"
    print(f"Loading in dist test statistics from {in_dist_test_path}")
    in_dist_test_statistics_file = np.load(in_dist_test_path)
    id_test_statistics = load_statistics(in_dist_test_statistics_file)
    score_test_id = best_gmm.score_samples(id_test_statistics)
    print('id test score: ', score_test_id.mean())

    # load out of distribution test dataset statistics
    out_of_dist_test_path = os.path.join(f"test_statistics_{args.model}_model/ddim{args.n_ddim_steps}", args.out_of_dist) + ".npz"
    print(f"Loading out of dist test statistics from {out_of_dist_test_path}")
    ood_test_statistics_file = np.load(out_of_dist_test_path)
    ood_test_statistics = load_statistics(ood_test_statistics_file)
    score_ood = best_gmm.score_samples(ood_test_statistics)
    print('ood test score: ', score_ood.mean())

    y_test_id = np.ones(score_test_id.shape[0])
    y_test_ood = np.zeros(score_ood.shape[0])
    y_true = np.append(y_test_id, y_test_ood)

    sample_score = np.append(score_test_id, score_ood)
    auroc = roc_auc_score(y_true, sample_score)

    print(f"In dist: {args.in_dist}, Out of dist: {args.out_of_dist}, DiffPath 6D AUROC: {auroc}")

    # Enhance train set with pseudo-id set
    pseudo_id_stats, ratio = get_pseudo_id_stats(id_test_statistics, score_test_id, ood_test_statistics, score_ood, -4)
    print('TP ratio: ', ratio)
    enhanced_train_stats = enhance_train_stats(id_train_statistics, pseudo_id_stats, args.enhance_ratio)
    print('Enhanced train stats: ', enhanced_train_stats.shape)
    best_gmm = gridsearch_and_fit(gmm_clf, param_grid, enhanced_train_stats)
    auroc = eval_auroc_with_stats(best_gmm, id_test_statistics, ood_test_statistics)
    print(f"After enhancement, DiffPath 6D AUROC: {auroc}")

if __name__ == "__main__":
    main()
