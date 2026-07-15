import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
def evaluate_model_by_group(
    df, 
    demographic_cols, 
    actual_col='actual', 
    predicted_col='predicted', 
    score_col='score'
):
    def compute_metrics(sub_df, label='', group='', group_val='', total_suicide=-1):
        if sub_df.empty:
            return {
                'Subset': label,
                'Group': group,
                'Group Value': group_val,
                'Patients': 0,
                # ranking scores
                'AUC-ROC': 0,
                'RMSE': 0,
                'TPR (%)': 0,
                'Precision (%)': 0,
                'TP %': 0,  # deprecated alias for TPR (%); kept for backward compatibility
                # classification scores
                'Accuracy': 0,
                'F1 Score': 0,
                'TP': 0,
                'TPR': 0,
                'FP': 0,
                'FPR': 0,
                'FN': 0,
                'FNR': 0
            }

        y_true = sub_df[actual_col]
        y_pred = sub_df[predicted_col]
        y_score = sub_df[score_col]

        # Confusion matrix: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = safe_confusion_matrix(y_true, y_pred)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        rmse = np.sqrt(mean_squared_error(y_true, y_score))

        # Ranking-based metrics at the top-k% threshold (when sub_df is a top-k% subset):
        #   TPR (top k%)       = actual positives in top k% / total positives in parent set
        #   Precision (top k%) = actual positives in top k% / size of top k% subset
        # For non-top-k subsets (Overall, per-group overall) these degenerate to 100%
        # and (positive rate), respectively.
        positives_in_subset = int(sum(y_true))
        tpr_top_pct = positives_in_subset * 100 / total_suicide if total_suicide > 0 else 0
        precision_top_pct = positives_in_subset * 100 / len(sub_df) if len(sub_df) > 0 else 0

        return {
            'Subset': label,
            'Group': group,
            'Group Value': group_val,
            'Patients': len(sub_df),
            # ranking scores (meaningful primarily at Top 1% / Top 5% risk subsets)
            'AUC-ROC': round(roc_auc_score(y_true, y_score), 3),
            'RMSE': round(rmse, 3),  # may not be applicable for classification tasks, but included for consistency
            'TPR (%)': round(tpr_top_pct, 3),
            'Precision (%)': round(precision_top_pct, 3),
            'TP %': round(tpr_top_pct, 3),  # deprecated alias for TPR (%); kept for backward compatibility
            # classification scores
            'Accuracy': round(accuracy_score(y_true, y_pred), 3),
            'F1 Score': round(f1_score(y_true, y_pred), 3),
            'TP': tp,
            'TPR': round(tpr, 3),
            'FP': fp,
            'FPR': round(fpr, 3),
            'FN': fn,
            'FNR': round(fnr, 3)
        }

    results = []
    total_suicide = sum(df[actual_col])

    # Overall (no stratification)
    results.append(compute_metrics(df, label='Overall', group='All', group_val='All', total_suicide=total_suicide))

    # Risk strata (Top 1% and 5%)
    df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    for pct in [0.01, 0.05]:
        top_n = max(int(len(df) * pct), 1)
        top_df = df_sorted.iloc[:top_n]
        results.append(compute_metrics(top_df, label=f'Top {int(pct*100)}% Risk', total_suicide=total_suicide))

    # Stratify by each demographic column
    for col in demographic_cols:
        for val in df[col].dropna().unique():
            df_group = df[df[col] == val]
            total_suicide = sum(df_group[actual_col])
            results.append(compute_metrics(df_group, label='Overall', group=col, group_val=val, total_suicide=total_suicide))

            df_sorted_group = df_group.sort_values(by=score_col, ascending=False)
            for pct in [0.01, 0.05]:
                top_n = max(int(len(df_group) * pct), 1)
                top_df_group = df_sorted_group.iloc[:top_n]
                results.append(compute_metrics(
                    top_df_group, 
                    label=f'Top {int(pct*100)}% Risk',
                    group=col,
                    group_val=val,
                    total_suicide=total_suicide
                ))

    return pd.DataFrame(results)



def safe_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        # Handle edge case explicitly
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (1, 1):
            # Single class (0 or 1)
            if y_true[0] == 0:
                tn = cm[0][0]
            else:
                tp = cm[0][0]
        elif cm.shape == (1, 2):
            tn, fp = cm[0]
        elif cm.shape == (2, 1):
            tn = cm[0][0]
            fn = cm[1][0]
        return tn, fp, fn, tp
    return cm.ravel()
