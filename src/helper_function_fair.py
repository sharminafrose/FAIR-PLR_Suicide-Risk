import numpy as np

def z_interact_multi_multi(X, group_dict, reference_group_dict):
    """
    X: feature matrix (n_samples, n_features)
    group_dict: dict of group_name -> group_vector (e.g., {"race": z1, "age": z2})
    reference_group_dict: dict of group_name -> reference_value

    usage:
    group_dict = {
        "race": [1, 2, 1, 3, 4, 2],
        "age": [2, 1, 3, 4, 5, 1]
    }
    ref_group_dict = {
        "race": 1,
        "age": 1
    }
    X_out = z_interact_multi_multi(X, group_dict, ref_group_dict)
    """
    X = np.c_[np.ones((X.shape[0], 1)), X]
    new_cols = [X]
    
    for group_name, z in group_dict.items():
        ref = reference_group_dict[group_name]
        unique_vals = sorted(set(z))
        
        for val in unique_vals:
            if val != ref:
                X_interact = np.copy(X)
                X_interact[np.array(z) != val] = 0
                new_cols.append(X_interact)

    return np.hstack(new_cols)


def pf_to_array(pf, num_feat, z, small_groups):
    """
    Example:
        Suppose you have 2 groups (0 and 1), 3 features per group, and you want to penalize group 0 more:
        
        pf = 0.1
        num_feat = 3
        z = [0, 1, 0, 1, 0]
        small_groups = [0]
        
        pf_to_array(pf, num_feat, z, small_groups)
        # Output: array([0.1, 0.1, 0.1, 1, 1, 1])
        # (First 3 values for group 0, next 3 for group 1)
    """
    pf_array = []
    single = type(pf) is not tuple
    for z_i in range(len(set(z))):
        if single:
            pf_array += [pf if z_i in small_groups else 1] * num_feat
        else:
            pf_array += [pf[small_groups.index(z_i)] if z_i in small_groups else 1] * num_feat
    return np.array(pf_array)

# Add interaction terms between the original features and each non-reference group (excluding group 0) to allow group-specific modeling (e.g., separate slopes for each group).
def z_interact_multi_group(X, z, most_prevalent_group):
    # example: X = [[1, 2],
    #               [3, 4],
    #               [5, 6]]
    #          z = [0, 1, 0]
    X = np.c_[np.ones((X.shape[0], 1)), X]   # Add bias/intercept column of 1s
    groups = sorted(list(set(z)))      # Get unique group labels
    new_cols = [X]                     # Start with original X (with intercept)
    for k in groups:
        if k != most_prevalent_group:
            X_interact = np.copy(X)
            X_interact[z != k] = 0     # Zero out rows not in group k
            new_cols.append(X_interact)    # Append interaction terms for group k
    # print(np.hstack(new_cols)[320])
    return np.hstack(new_cols)
    # example return : [[1, 1, 2, 0, 0, 0],
    #                   [1, 3, 4, 1, 3, 4],
    #                   [1, 5, 6, 0, 0, 0]]
    # coff = [itercept, coff_1, coff_2, itercept, coff_4, coff_5]
    # coff = [itercept, coff_1, coff_2, --> female/male (base)
    #         itercept, coff_4, coff_5] --> male




def get_n_k(z_train_curr, size_weighting):
    """Compute per-observation subgroup-size weights for FAIR-PLR.

    With size_weighting=True, returns w_i = n_min / n_{k(i)}, where n_{k(i)}
    is the count of observations in the subgroup that observation i belongs
    to, and n_min is the smallest subgroup count. This minority-upweighting
    scheme ensures every subgroup contributes equal aggregate mass to the
    weighted loss: the smallest subgroup receives weight 1.0 and larger
    subgroups receive fractional weights in (0, 1].

    Example:
        z_train_curr = [0, 1, 0, 1, 1, 2]
        counts = {0: 2, 1: 3, 2: 1}, n_min = 1
        returns [0.5, 1/3, 0.5, 1/3, 1/3, 1.0]

    Args:
        z_train_curr: array-like of subgroup labels.
        size_weighting: if True, return inverse-frequency weights; otherwise,
            return a uniform weight vector of ones.

    Returns:
        numpy array of per-observation weights, same length as z_train_curr.
    """
    if size_weighting:
        values, counts = np.unique(z_train_curr, return_counts=True)
        count_dict = dict(zip(values, counts))
        min_count = min(counts)
        curr_n_k = np.array([min_count / count_dict[z_i] for z_i in z_train_curr])
    else:
        curr_n_k = np.ones(len(z_train_curr))
    return curr_n_k


def get_fair_plus_survey_weights(z_fair, survey_weights, n_pool_years=1):
    """Combine NSDUH survey weights with FAIR subgroup balancing via sequential correction.

    Implements the combined weight derived in Methods (Eq. star):
        w_i = (s_i / Y) * min_k(n_hat_k) / n_hat_{k(i)}
    where s_i is the NSDUH person-level analysis weight (ANALWT_C for 2013-2019,
    ANALWTQ1Q4_C for 2020, or ANALWT2_C for 2021-2023), Y = n_pool_years is the
    number of years pooled (SAMHSA's canonical multi-year rescaling divisor), and
    n_hat_k = sum of rescaled survey weights within FAIR subgroup k.

    The combined weight simultaneously
        (C1) equalizes aggregate loss contribution across FAIR subgroups,
        (C2) preserves within-subgroup relative sampling weights, and
        (C3) eliminates Y-fold population double-counting when pooling Y years.
    Pass n_pool_years=11 when training on the full 2013-2023 pooled dataset;
    pass n_pool_years=1 (default) for single-year / homogeneous-window analyses
    such as the 2021-2023 survey-weight sensitivity check.

    Args:
        z_fair: array of FAIR subgroup labels (e.g., Age, Race, Sex).
        survey_weights: array of NSDUH sample weights, one per respondent.
        n_pool_years: number of pooled survey years (SAMHSA divisor Y).

    Returns:
        Combined weight array to pass as `curr_n_k` to FairElasticGlmNet.fit().
    """
    w_survey = np.asarray(survey_weights, dtype=float) / float(n_pool_years)
    z_fair = np.asarray(z_fair)
    effective = {k: w_survey[z_fair == k].sum() for k in np.unique(z_fair)}
    target = min(effective.values())
    w_corr = np.array([target / effective[z_i] for z_i in z_fair])
    return w_survey * w_corr
