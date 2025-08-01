import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import numpy as np


def load_subset_data(loader: DataLoader, subset_ratio: float):
    '''
    Returns a new dataloader loading a random subset from the original dataloader.

    Params:
        loader: torch.utils.data.DataLoader, the original dataloder containing whole dataset
        subset ratio: float, the ratio of subset length to dataset length

    Returns:
        torch.utils.data.DataLoader

    Examples:
        from ood_utils import load_data
        cifar_loader = load_data(args.dataset, args.data_dir, args.batch_size, args.image_size, train=True)
        cifar_subset_loader = subset_loader(cifar_loader, 0.1)
    '''
    num_samples = len(loader.dataset)
    subset_samples = int(subset_ratio * num_samples)
    indices = torch.randperm(num_samples).numpy()[:subset_samples]
    sampler = SubsetRandomSampler(indices)
    return DataLoader(
        dataset=loader.dataset,
        num_workers=loader.num_workers,
        batch_size=loader.batch_size,
        shuffle=False,
        sampler=sampler
    )

def get_subset_stats(stats, num_samples: int):
    if isinstance(stats, dict):
        tot_samples = stats['deps_dt_sq_sqrt'].shape[0]
    else:
        tot_samples = stats.shape[0]
    tot_samples = stats['deps_dt_sq_sqrt'].shape[0]
    print('tot samples: ',tot_samples)
    indices = torch.randperm(tot_samples).numpy()[:num_samples]
    return stats[indices] if isinstance(stats, np.ndarray) \
                          else {k: v[indices] for k, v in stats.items()}

def get_pseudo_id_stats(id_test_stats: np.ndarray, score_id: np.ndarray,
                        ood_test_stats: np.ndarray, score_ood: np.ndarray,
                        threshold_score: float):
    '''
    Extracts pseudo-id samples from test set and calculates the percentage of true positive samples in the set.

    Params:
        id_test_stats: np.ndarray of shape (N, 6), the computed stats of id test set
        score_id: np.ndarray of shape (N,), the score of id test set
        ood_test_stats: np.ndarray of shape (N, 6), the computed stats of ood test set
        score_ood: np.ndarray of shape (N,), the score of ood test set
        threshold_score: float, decision boundary of id vs ood

    Returns:
        (pseudo-id stats, percentage of true positive samples)

    '''
    tp_indices = np.where(score_id >= threshold_score)[0]
    print('num tp samples: ', tp_indices.shape[0])
    fp_indices = np.where(score_ood >= threshold_score)[0]
    print('num fp samples: ', fp_indices.shape[0])
    pseudo_id_stats = np.concatenate([id_test_stats[tp_indices], ood_test_stats[fp_indices]], axis=0)
    ratio = len(tp_indices) / len(pseudo_id_stats)
    return (pseudo_id_stats, ratio)


def enhance_train_stats(train_stats: np.ndarray, 
                        pseudo_id_stats: np.ndarray, 
                        num_enhance_samples: int=0) -> np.ndarray:
    '''
    Enhance training set with pseudo-id data in the test sets. Returns the 
    enhanced training set stats.

    Param:
        <train_stats>: np.ndarray of shape (N, 6), the computed stats of 
        training set
        <pseudo_id_stats>: np.ndarray of shape (N, 6), the estimated id 
        set in test set
        <num_enhance_samples>: int, the number of pseudo-id samples to be 
        added into training set

    Returns:
        enhanced train stats, np.ndarray
    '''
    num_pseudo_id_samples = pseudo_id_stats.shape[0]
    indices = torch.randperm(num_pseudo_id_samples).numpy()[:num_enhance_samples]
    enhance_samples = pseudo_id_stats[indices, :]
    return np.concatenate([train_stats, enhance_samples], axis=0)

def get_topk_score_indices(id_score: np.ndarray, 
                           ood_score: np.ndarray, 
                           topk: int, 
                           use_greatest=True) -> tuple[np.ndarray, np.ndarray]:
    '''
    Given two score arrays, get the indices of greatest or lowest top-k 
    scores of the merged array in the original arrays. 

    Params: 
        <id_score>: np.ndarray of shape N_1, the ID score array.
        <ood_score>: np.ndarray of shape N_2, the OOD score array.
        <topk>: int, number of samples needed
        <use_greatest>: bool, whether return the greatest samples or 
        the lowest samples. 
    
    Return: 
        A tuple of two ndarrays (<tp_indices>, <fp_indices>). The indices 
        of top-k samples in the <id_score> array and <ood_score> array
    '''
    num_id_samples = id_score.shape[0]
    concat_score = np.concatenate([id_score, ood_score], axis=0)
    topk_indices = np.argsort(concat_score)[-topk:] if use_greatest \
                   else np.argsort(concat_score)[:topk]
    tp_indices = topk_indices[np.where(topk_indices < num_id_samples)[0]]
    fp_indices = topk_indices[np.where(topk_indices >= num_id_samples)[0]] % num_id_samples
    return tp_indices, fp_indices

def get_topk_score_stats(id_test_stats: np.ndarray, id_score: np.ndarray,
                         ood_test_stats: np.ndarray, ood_score: np.ndarray,
                         topk: int, use_greatest=True) -> tuple[np.ndarray, 
                                                                np.ndarray, 
                                                                np.ndarray, 
                                                                float]:
    '''
    Given two stats datasets and two score arrays of the stats datasets, return 
    the stats samples of top-k greatest/lowest score, their corresponding score, 
    label and ratio of TP/(TP+FP)

    Params: 
        <id_test_stats>: np.ndarray of shape (N1, D), the stats of ID dataset 
        <id_score>: np.ndarray of shape (N1, ), the score of ID dataset samples
        <ood_test_stats>: np.ndarray of shape (N2, D), the stats of OOD dataset
        <ood_score>: np.ndarray of shape (N2, ), the score of OOD dataset samples 
        <topk>: int, the number of samples to be returned
        <use_greatest>: bool: return the greatest samples or not
    
    Returns:
        <pseudo_stats>: np.ndarray of shape (topk, D), the top-k stats samples 
        <pseudo_score>: np.ndarray of shape (topk,), the score of selected samples
        <labels>: np.ndarray of shape (topk,), the labels of selected samples
        <ratio>: float, the ratio of TP/(TP + FP)
    '''
    tp_indices, fp_indices = get_topk_score_indices(id_score, ood_score, topk, use_greatest)
    tp_score = id_score[tp_indices]
    fp_score = ood_score[fp_indices]
    tp_stats = id_test_stats[tp_indices]
    fp_stats = ood_test_stats[fp_indices]
    pseudo_stats = np.concatenate([tp_stats, fp_stats], axis=0)
    pseudo_score = np.concatenate([tp_score, fp_score], axis=0)
    labels = np.zeros_like(pseudo_score, dtype=np.int32)
    num_id_samples = tp_indices.shape[0]
    labels[:num_id_samples] = 1
    ratio = tp_indices.shape[0] / pseudo_stats.shape[0]
    return pseudo_stats, pseudo_score, labels, ratio

def merge_stats_with_indices(stats1: np.ndarray, 
                             indices1: np.ndarray, 
                             stats2: np.ndarray, 
                             indices2: np.ndarray) -> np.ndarray: 
    '''
    Given two stats datasets and two arrays of indices, merge the selected
    samples into an ndarray and shuffle.

    Params: 
        <stats1>: np.ndarray of shape (N1, D) 
        <indices1>: np.ndarray of shape (N1,)
        <stats2>: np.ndarray of shape (N2, D)
        <indices2>: np.ndarray of shape (N2)

    Return: 
        Merged ndarray after shuffle
    '''
    s1, s2 = [], []
    if indices1.shape[0] > 0 and stats1.shape[0] > 0:
        s1 = stats1[indices1]
    if indices2.shape[0] > 0 and stats2.shape[0] > 0:
        s2 = stats2[indices2]
    s = np.concatenate([s1, s2], axis=0)
    return np.random.permutation(s)