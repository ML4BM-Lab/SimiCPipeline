#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This is an implementation of simicLASSO
MODIFIED by IRENE MARIN to incorporate in new Dockerfile
"""
import pandas as pd
import numpy as np
import time
import random
import copy
import itertools
import sys
from pathlib import Path
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.linalg import eigh
from simicpipeline.utils.io import write_pickle


# Data extraction functions
def extract_cluster_from_assignment(df_in, assignment):
    '''
    Given DataFrame with all cells and all genes,
    provided a dictionary with {label: df_cluster}
    where df_cluster are selected by rows
    '''
    cluster_dict = {}
    for label in set(assignment): 
        cluster_idx = np.where(assignment == label)
        tmp = df_in.index.values
        df_cluster_index = tmp[cluster_idx]
        cluster_dict[label] = df_in.loc[df_cluster_index]
    return cluster_dict

def extract_tf_target_mat_from_cluster_dict(cluster_dict, tf_list, target_list):
    '''
    Given a dict with { label: df_cluster},
    extract the TF matrix and target gene matrix for each cluster.
    Args:
        - cluster_dict is the input dictionary, with { label: df_cluster}
         -target_list is the input target gene list.
        - tf_list is the input TF list.
    Returns:
    mat_dict is a dictionary:
        { label:
            { 'tf_mat': np.matrix,
              'target_mat': np.matrix}
        }
    TF_list is the matching list, in case tf_list contains TFs not in df columns.
    Target_list is the matching list, in case target_list contains genes not in df columns.
    They should not differ as we already filtered tf_list and target_list before, but for reusability of the function.
    '''
    mat_dict = {}
    for label in cluster_dict:
        cur_df = cluster_dict[label]    
        TF_df = cur_df[tf_list]
        tmp_mat = TF_df.values
        # add fixed value 1 to last column, for the bias term in regression.
        m, n_x = tmp_mat.shape
        TF_mat = np.ones((m, n_x + 1))
        TF_mat[:, :-1] = tmp_mat
        TF_list = TF_df.columns.values.tolist()
        target_df = cur_df[target_list]
        target_mat = target_df.values
        Target_list = target_df.columns.values.tolist()
        sys.stdout.flush()
        print('cell type ', label)
        print('\tTF size:', TF_mat.shape)
        print('\tTarget size:', target_mat.shape)
        mat_dict[label] = { 'tf_mat':TF_mat,
                            'target_mat':target_mat}
    return mat_dict, TF_list, Target_list

def load_mat_dict_and_ids(df_in, tf_list, target_list, assignment):
    '''
    Get clustered data matrix, as well as the TF_ids, and Target genes ids found in the data matrix
    '''
    cluster_dict = extract_cluster_from_assignment(df_in, assignment)
    mat_dict, TF_ids, Target_ids = extract_tf_target_mat_from_cluster_dict(cluster_dict,
            tf_list, target_list)
    return mat_dict, TF_ids, Target_ids

# Regression function and metrics
def loss_function_value(mat_dict, weight_dict, similarity, lambda1, lambda2):
    loss = 0
    num_labels = len(weight_dict.keys())
    for label in mat_dict.keys():
        Y_i = mat_dict[label]['target_mat']
        X_i = mat_dict[label]['tf_mat']
        W_i = weight_dict[label]
        m, n_x = X_i.shape
        loss += 1/m * (np.linalg.norm(Y_i - X_i @ W_i) ** 2)
        loss += lambda1 * np.linalg.norm(W_i, 1)
        
        if similarity:
            if label != max(mat_dict.keys()):
                W_ip1 = weight_dict[label + 1]
            else:
                W_ip1 = W_i
            loss += lambda2 * (np.linalg.norm(W_i - W_ip1) ** 2)
    return loss

def std_error_per_cluster(mat_dict, weight_dict):
    std_error_dict = {}
    for label in mat_dict.keys():
        Y_i = mat_dict[label]['target_mat']
        X_i = mat_dict[label]['tf_mat']
        W_i = weight_dict[label]
        m, n_x = X_i.shape
        std_error = np.sqrt(1/m) * np.linalg.norm( Y_i - X_i @ W_i, axis = 0)
        std_error_dict[label] = std_error
    return std_error_dict

def get_gradient(mat_dict, weight_dict, label, similarity, lambda1, lambda2):
    '''
    get graident of loss function
    of W_k, weight matrix for cluster/label k
    '''
    Y_i = mat_dict[label]['target_mat']
    X_i = mat_dict[label]['tf_mat']
    W_i = weight_dict[label]
    m, n_x = X_i.shape

    last_label = max(weight_dict.keys())
    first_label = min(weight_dict.keys())

    grad_f = 2/m * X_i.T @ (X_i @ W_i - Y_i) + lambda1 * np.sign(W_i)
    if similarity:
        # [0, 1, 2], pick 0, then (0 -1) % 3 = 2, the last term
        # if pick 2, then (2+1) % 3 = 0, the first term
        if label == last_label:
            W_i_plus1 = W_i
        else:
            W_i_plus1 = weight_dict[(label + 1)]
        if label == first_label:
            W_i_minus1 = W_i
        else:
            W_i_minus1 = weight_dict[(label -1)]
        grad_f += 2 * lambda2 * (W_i - W_i_plus1 + W_i - W_i_minus1)
    return grad_f

def get_L_max(mat_dict, similarity, lambda2):
    '''
    Calculate the Lipschitz constant for each cluster/label
    '''
    L_max_dict = {}
    for label in mat_dict.keys():
        Y_i = mat_dict[label]['target_mat']
        X_i = mat_dict[label]['tf_mat']
        m, n_x = X_i.shape
        #### eigvals = (lo, hi) indexes of smallest and largest (in ascending order) Irene Note: since v1.5.0 scypy argument changed name `subset_by_index`
        #### eigenvalues and corresponding eigenvectors to be returned.
        #### 0 <= lo < hi <= M -1 
        L_tmp = 2/m * eigh(X_i.T @ X_i, subset_by_index = (n_x - 1, n_x - 1), eigvals_only = True)
        if similarity:
            L_tmp += 4 * lambda2
        L_max_dict[label] = L_tmp
    return L_max_dict

def get_r_squared(y_true, y_pred, k = 0,  
        sample_weight = None,
        multioutput='uniform_average'):
    '''
    compute R2 score for regression model,
    if y is a matirx, (multiple y_vector), then first compute R2 score for
    each column, and get average of column R2 score.
    otherwise, see multioutput explaination in sklearn.
    k: number of independent variable in X
        k = 0, ordinary R2, nothing changed
        k = X.shape[1], Adjusted R-Squared, 
    '''
    R2_val = r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput = multioutput)
    num_sample = y_pred.shape[0]
    assert num_sample > 1
    adj_R2 = 1 - (1 - R2_val) * ((num_sample - 1)/(num_sample - k - 1))
    return adj_R2

def average_r2_score(mat_dict, weight_dict):
    '''
    Calculate the total avereage R2 squared score for all clusters
    i.e.
        average_r2 = sum(r2 for each cluster) / num_cluster
    '''
    num_cluster = len(weight_dict.keys())
    sum_r2 = 0
    r2_dict = {}
    for label in weight_dict:
        X_i = mat_dict[label]['tf_mat']
        Y_i = mat_dict[label]['target_mat']
        W_i = weight_dict[label]
        m, n_x = X_i.shape
        Y_pred = X_i @ W_i
        # # ordinary R2
        # sum_r2 += get_r_squared(Y_i, Y_pred, k)
        # adjusted R2
        W_i_avg = np.mean(W_i, axis = 1)
        W_avg_count = np.count_nonzero(W_i_avg > 1e-3)
        num_idpt_variable = min(m, W_avg_count) - 2
        list_of_r_squared = get_r_squared(Y_i, Y_pred, k = num_idpt_variable,
                multioutput = 'raw_values')
        r2_dict[label] = list_of_r_squared
        sum_r2 += np.mean(list_of_r_squared)
    aver_r2 = sum_r2 / num_cluster
    return aver_r2, r2_dict

# Main RCD function
def rcd_lasso_multi_cluster(mat_dict, similarity,
        lambda1 = 1e-3, lambda2 = 1e-3,
        slience = False, max_rcd_iter = 50000):
    L_max_dict = get_L_max(mat_dict, similarity, lambda2)
    weight_dict = {}
    # initialize weight dict for each label/cluster
    for label in mat_dict.keys():
        _, n_x = mat_dict[label]['tf_mat'].shape
        m, n_y = mat_dict[label]['target_mat'].shape
        tmp = np.zeros((n_x, n_y))
        weight_dict[label] = tmp
    weight_dict_0 = copy.deepcopy(weight_dict)
    loss_0 = loss_function_value(mat_dict, weight_dict, similarity, lambda1, lambda2)
    r2_0, r2_dict_0 = average_r2_score(mat_dict, weight_dict)
    if not slience:
        sys.stdout.flush()
        print('start RCD process...')
        print('-' * 7)
        print('\tloss w. reg before RCD: {:.4f}'.format(loss_0))
        print('\tR squared before RCD: {:.4f}'.format(r2_0))
        print('-' * 7)
    num_iter = 0
    label_list = list(mat_dict)
    time_sum = 0
    pause_step = 50000
    while num_iter <= max_rcd_iter:
        num_iter += 1
        if num_iter % pause_step == 0 and not slience:
            t1 = time.time()
            loss_tmp = loss_function_value(mat_dict, weight_dict, similarity, lambda1, lambda2)
            t2 = time.time()
            sys.stdout.flush()
            print('\ttime elapse in eval: {:.4f}s'.format(t2 - t1))
            print('\ttime elapse in update: {:.4f}s'.format(time_sum / pause_step))
            time_sum = 0
            print('\titeration {}, loss w. reg = {:.4f}'.format(num_iter, loss_tmp))
            print('-' * 7)
        t1 = time.time()
        # In this implementation we randomly select a cluster to update the weights in the current iteration, while leaving the weights for other clusters unchanged.
        # This introduces stochasticity into the optimization process that i) can help the algorithm escape poor local minima ii) avoid biases introduced by a fixed update order and iii) can sometimes lead to faster convergence in practice.
        label = random.choice(label_list)
        step_size = 1 / L_max_dict[label]
        grad_f = get_gradient(mat_dict, weight_dict, label, similarity, lambda1, lambda2)
        weight_dict[label] -= step_size * grad_f
        t2 = time.time()
        time_sum += (t2 - t1)
    loss_final = loss_function_value(mat_dict, weight_dict, similarity, lambda1, lambda2)
    r2_final, r2_dict_final = average_r2_score(mat_dict, weight_dict)
    if not slience:
        sys.stdout.flush()
        print('\tloss w. reg after RCD: {:.4f}'.format(loss_final))
        print('\tR squared after RCD: {:.4f}'.format(r2_final))
        print('-' * 7)
        print('Done RCD process!')
    return weight_dict, weight_dict_0

# Cross validation functions
def cross_validation(mat_dict_train, similarity, list_of_l1, list_of_l2, k, max_rcd_iter):
    '''
    Perform 5-fold cross validation on training set
    '''
    opt_r2_score = float('-inf')
    for lambda1, lambda2 in itertools.product(list_of_l1, list_of_l2):
        # run k-fold evaluation
        aver_r2_tmp, r2_tmp = k_fold_evaluation(k, mat_dict_train, similarity, lambda1, lambda2, max_rcd_iter)
        sys.stdout.flush()
        print('lambda1 = {}, lamda2 = {}, done'.format(lambda1, lambda2))
        print('----> Averaged adjusetd R2 = {:.4f}'.format(aver_r2_tmp))
        print(f"----> R2 in folds = [{', '.join(f'{x:.4f}' for x in r2_tmp)}]")
        print('////////')
        sys.stdout.flush()
        if aver_r2_tmp > opt_r2_score:
            l1_best, l2_best = lambda1, lambda2
            opt_r2_score = aver_r2_tmp
    return l1_best, l2_best, opt_r2_score

def k_fold_evaluation(k, mat_dict_train, similarity, lambda1, lambda2, max_rcd_iter):
    '''
    split training data set into equal k fold.
    train on (k - 1) and test on rest.
    r2_tmp = average r2 from each evalutaion
    '''
    r2_tmp = 0
    r2_aver_list = []
    for idx in range(k):
        mat_dict_train, mat_dict_eval = get_train_mat_in_k_fold(mat_dict_train, idx, k)
        weight_dict_trained, _ = rcd_lasso_multi_cluster(mat_dict_train, similarity, lambda1, lambda2, slience = True, max_rcd_iter = max_rcd_iter)
        r2_aver, _ = average_r2_score(mat_dict_eval, weight_dict_trained)
        r2_aver_list.append(r2_aver)
        r2_tmp += r2_aver
    aver_r2_tmp = r2_tmp/k
    return aver_r2_tmp, r2_aver_list

def get_train_mat_in_k_fold(mat_dict, idx, k):
    '''
    split input: mat_dict
    into k folder, select idx as test, rest as train
    '''
    mat_dict_train, mat_dict_eval = {}, {}
    for label in mat_dict:
        X_i = mat_dict[label]['tf_mat']
        Y_i = mat_dict[label]['target_mat']
        _, n_x = X_i.shape

        concat_XY = np.hstack((X_i, Y_i))
        split_XY = np.array_split(concat_XY, k)
        eval_x, eval_y = split_XY[idx][:, :n_x], split_XY[idx][:, n_x:]
        mat_dict_eval[label] = {
                'tf_mat': eval_x,
                'target_mat': eval_y
                }

        del split_XY[idx]
        tmp_XY = np.vstack(split_XY)
        train_x, train_y = tmp_XY[:, :n_x], tmp_XY[:, n_x:]
        mat_dict_train[label] = {
                'tf_mat': train_x,
                'target_mat': train_y
                }
    return mat_dict_train, mat_dict_eval

# Main function to be called
def simicLASSO_op(*,  # Force all arguments to be keyword-only
        p2df: str,
        p2tf: str,
        p2assignment: Optional[str] = None,
        p2saved_file: str,
        df_with_label: bool = False,
        similarity: bool = True,
        lambda1: float = 1e-2,
        lambda2: float = 1e-5,
        max_rcd_iter: int = 500000,
        cross_val: bool = False,
        k_cv: int = 5,
        max_rcd_iter_cv: int = 10000,
        num_rep: int = 1,
        list_of_l1: Optional[List[float]] = None,
        list_of_l2: Optional[List[float]] = None 
) -> None:
    '''
    perform the GRN inference algorithm, simicLASSO.
    Args:
        p2df (str): path to dataframe
            dataframe should be like (with optional label column if p2assignment is not provided):
                    gene1, gene2, ..., genek, label
            cell1:  x,     x,   , ..., x,   , type1
            cell2:  x,     x,   , ..., x,   , type2
        p2tf (str): path to TF list file / path to list of all TFs.
        p2assignment (str): path to clustering order assignment file.
                      a text file with each line corresponding to cell order.
        p2saved_file (str): path to save the output weight dictionary.
        df_with_label (bool): Whether the dataframe contains a label column with assignment labels.
        similarity (bool): Enables similarity constraint for Lipswitz constant (RCD process).
        lambda1 (float): L1 regularization parameter (sparsity).
        lambda2 (float): L2 regularization parameter (network similarity).
        max_rcd_iter (int): Maximum RCD iterations.
        cross_val (bool): Whether to perform cross-validation to select optimal lambdas.
        k_cv (int): Number of folds for cross-validation.
        max_rcd_iter_cv (int): Maximum number of RCD iterations in the cross-validation step.
        num_rep (int): Number of repetitions for test evaluation.
        list_of_l1 (list): List of L1 values for cross-validation.
        list_of_l2 (list): List of L2 values for cross-validation.
    Returns: 
        save the weight dictionary and gene list to p2saved_file.
    '''
    p2df = Path(p2df)
    if p2df.suffix == '.pickle':
        original_df = pd.read_pickle(p2df)
    elif p2df.suffix == '.csv':
        original_df = pd.read_csv(p2df, index_col = 0, header = 0)
    else:
        raise ValueError('Dataframe file format not supported! Please provide a .pickle or .csv file.')
    
    if df_with_label:
        if 'label' not in original_df.columns:
            raise ValueError("Dataframe is indicated to contain label column, but 'label' column not found in dataframe!")
        label_idx = list(original_df.columns).index('label')
        feat_cols = list(original_df.columns)
        feat_cols.pop(label_idx)

        assignment = list(original_df['label'].values)
        df = original_df.loc[:,feat_cols]
    else:
        feat_cols = list(original_df.columns)
        df = original_df
    
    if p2assignment != None:
        assignment_df = pd.read_csv(p2assignment, header = None)
        assignment = assignment_df[0].values

    #### BEGIN of the regression part
    sys.stdout.flush()
    print('Expression matrix for regression shape = ', df.shape)
    X_train, X_test, y_train, y_test = train_test_split(df, assignment, test_size = 0.2, random_state = 1)
    
    sys.stdout.flush()
    print('df test = ', X_test.shape)
    print('test data assignment set:', set(list(y_test)))
    print('df train = ', X_train.shape)
    print('train data assignment set:', set(list(y_train)))
    print('-' * 7)
    
    # Read TF list
    p2tf = Path(p2tf)
    if p2tf.suffix == '.pickle':
        tf_list = pd.read_pickle(p2tf)
    elif p2tf.suffix == '.csv':
        tf_df = pd.read_csv(p2tf, header = None)
        tf_list = tf_df.iloc[:,0].tolist()
    else:
        raise ValueError('TF list file format not supported! Please provide a .pickle or .csv file.')
    
    if isinstance(tf_list, pd.DataFrame):
        tf_list = tf_list.iloc[:,0].tolist()
    # Check that tf_list contains tfs found in df columns
    tf_list = [x for x in tf_list if x in feat_cols]
    if len(tf_list) == 0:
        raise ValueError('No TFs found in the dataframe columns! Please check the TF list file and dataframe file.')
    # Select target genes as the remaining genes not in tf_list
    target_list = [x for x in feat_cols if x not in tf_list]
    sys.stdout.flush()
    print('-' * 7)

    print('.... generating train set')
    mat_dict_train, TF_ids, Target_ids = load_mat_dict_and_ids(X_train, tf_list, target_list, y_train)
    print('-' * 7)

    print('.... generating test set')
    mat_dict_test, _, _ = load_mat_dict_and_ids(X_test, tf_list, target_list, y_test)
    print('-' * 7)

    if cross_val == True:
        ### run cross_validation!!!!!! #############
        sys.stdout.flush()
        print("\n" + "-"*70)
        print('Sart cross validation!!!')
        print("-"*70 + "\n")
        print(f"Trying lambda1 = {list_of_l1} and lambda2 = {list_of_l2}")
        l1_opt, l2_opt, r2_opt = cross_validation(mat_dict_train, similarity, list_of_l1, list_of_l2, k_cv, max_rcd_iter_cv)
        sys.stdout.flush()
        print("Criss Validation done! \n")
        print('Selected: lambda1 = {}, lambda2 = {}, opt R squared on eval {:.4f}'.format(l1_opt, l2_opt, r2_opt))
        print('-' * 70)
        sys.stdout.flush()
        lambda1 = l1_opt
        lambda2 = l2_opt
        ############### end of cv ####################

    #### optimize using RCD
    print("\n" + "-"*70)
    print('Begin the SimiC Regression!!!')
    print("-"*70 + "\n")
    train_error, test_error = [], []
    r2_final_train, r2_final_test = [], []
    r2_final_0 = 0
    for _ in range(num_rep):
        trained_weight_dict, weight_dict_0 = rcd_lasso_multi_cluster(mat_dict_train, similarity,
                                    lambda1, lambda2, slience = True, max_rcd_iter = max_rcd_iter)
        test_error.append(loss_function_value(mat_dict_test, trained_weight_dict, similarity,
                lambda1 = 0, lambda2 = 0))
        train_error.append(loss_function_value(mat_dict_train, trained_weight_dict, similarity,
                lambda1 = 0, lambda2 = 0))
        std_error_dict_test = std_error_per_cluster(mat_dict_test, trained_weight_dict)
        # Save r2 metrics
        r2_final_train.append(average_r2_score(mat_dict_train, trained_weight_dict)[0])
        r2_aver_test, r2_dict_test = average_r2_score(mat_dict_test, trained_weight_dict)
        r2_final_test.append(r2_aver_test)
        r2_final_0 += average_r2_score(mat_dict_test, weight_dict_0)[0]   
    sys.stdout.flush()
    print('-' * 7)
    print('final train error w.o. reg = {:.4f} +/- {:.4f} SD'.format(np.mean(train_error), np.std(train_error)))
    print('test error w.o. reg = {:.4f} +/- {:.4f} SD'.format(np.mean(test_error), np.std(test_error)))
    print('-' * 7)
    print('R squared of test set (before): {:.4f}'.format(r2_final_0/num_rep))

    print('R squared of train set (after): {:.4f} +/- {:.4f} SD'.format(np.mean(r2_final_train), np.std(r2_final_train)))
    print('R squared of test set (after): {:.4f} +/- {:.4f} SD'.format(np.mean(r2_final_test), np.std(r2_final_test)))

    dict_to_saved = {'weight_dic' : trained_weight_dict,
                     'adjusted_r_squared': r2_dict_test,
                     'standard_error': std_error_dict_test,
                     'TF_ids'     : [symbols for symbols in TF_ids],
                     'query_targets' : [symbols for symbols in Target_ids]
                     }

    write_pickle(dict_to_saved, p2saved_file)

