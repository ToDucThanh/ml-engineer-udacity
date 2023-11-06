import matplotlib.pyplot as plt
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier


def plot_feature_importance(X_train, y_train, 
                            hyperparameters={
                                "n_estimators": 100,
                                "random_state": 42
                            }
                           ):
    """Plot feature importance figure from Random Forest Classifier
    Args:
        X_train, y_train: train set
        hyperparameters: hyperparameters of Random Forest Classifier. 
                         Defaults: {"n_estimators": 100, "random_state": 42}
    """
    
    rf = RandomForestClassifier(**hyperparameters)
    rf.fit(X_train, y_train)
    pi = permutation_importance(
        rf, 
        X_train, 
        y_train, 
        n_repeats=10,
        random_state=42)
    
    pi_sorted_idx = pi.importances_mean.argsort()
    tree_importance_sorted_idx = np.argsort(rf.feature_importances_)
    tree_idx = np.arange(0, len(rf.feature_importances_)) + 0.5

    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    ax1.barh(tree_idx,
             rf.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticklabels(X_train.columns[tree_importance_sorted_idx])
    ax1.set_yticks(tree_idx)
    ax1.set_ylim((0, len(rf.feature_importances_)))
    ax2.boxplot(pi.importances[pi_sorted_idx].T, 
                vert=False,
                labels=X_train.columns[pi_sorted_idx])
    
    fig.tight_layout()
    plt.show()