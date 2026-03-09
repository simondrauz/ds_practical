"""
GAM (Generalized Additive Model) analysis for trajectory prediction metrics.
Provides interpretable variable effects on prediction quality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f, te, GammaGAM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


def plot_correlation_matrix(df: pd.DataFrame, feature_cols: list, save_dir: str = None):
    """Plot correlation matrix and flag highly correlated pairs."""
    corr = df[feature_cols].corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                annot_kws={'size': 8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / 'correlation_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()

    # Flag highly correlated pairs (|r| > 0.7)
    threshold = 0.7
    print(f"\nHighly correlated pairs (|r| > {threshold}):")
    print(f"{'Feature 1':<30} {'Feature 2':<30} {'Correlation':<12}")
    print("-" * 72)
    flagged = set()
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                print(f"{feature_cols[i]:<30} {feature_cols[j]:<30} {r:<12.3f}")
                flagged.add(feature_cols[j])  # flag the second one for potential removal

    if flagged:
        print(f"\nConsider removing these redundant features: {sorted(flagged)}")
    else:
        print("\nNo highly correlated pairs found.")

    return corr, flagged

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the evaluation metrics CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nBasic stats:\n{df.describe()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


def identify_columns(df: pd.DataFrame):
    """
    Identify feature columns vs target columns.
    Adjust this based on your actual CSV structure.
    """
    # Common trajectory metric targets (adjust to your data)
    potential_targets = [
        'ade', 'fde', 'minADE', 'minFDE', 'min_ade', 'min_fde',
        'ADE', 'FDE', 'miss_rate', 'MR', 'brier_fde', 'nll',
        'avg_ade', 'avg_fde', 'final_displacement_error',
        'average_displacement_error'
    ]
    
    # Find which target columns exist
    targets = [c for c in df.columns if c in potential_targets]
    
    # Common ID/metadata columns to exclude from features


    exclude_patterns = [
        'instance', 'id', 'idx', 'name', 'type'
    ]


    all_cols = df.columns.tolist()
    feature_cols = []
    categorical_cols = []
    
    for col in all_cols:
        if col in targets:
            continue
        if any(pat in col.lower() for pat in exclude_patterns):
            continue
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            feature_cols.append(col)
    
    print(f"\nIdentified targets: {targets}")
    print(f"Identified numeric features: {feature_cols}")
    print(f"Identified categorical features: {categorical_cols}")
    
    return feature_cols, categorical_cols, targets


def prepare_data(df: pd.DataFrame, feature_cols: list, categorical_cols: list, 
                 target_col: str):
    """Prepare data for GAM fitting."""
    # Drop rows with NaN in relevant columns
    relevant_cols = feature_cols + categorical_cols + [target_col]
    df_clean = df[relevant_cols].dropna()
    print(f"\nAfter dropping NaN: {len(df_clean)} rows (dropped {len(df) - len(df_clean)})")
    
    # Encode categoricals
    cat_mappings = {}
    for col in categorical_cols:
        df_clean[col + '_encoded'], mapping = pd.factorize(df_clean[col])
        cat_mappings[col] = mapping
        feature_cols_final = [c for c in feature_cols] + [col + '_encoded' for col in categorical_cols]
    
    if not categorical_cols:
        feature_cols_final = feature_cols
    
    X = df_clean[feature_cols_final].values
    y = np.log(df_clean[target_col].values + 1e-6)
    
    return X, y, feature_cols_final, cat_mappings, df_clean


def fit_gam(X: np.ndarray, y: np.ndarray, feature_names: list, 
            categorical_encoded: list = None, n_splines: int = 20,
            lam: float = 0.6):
    """
    Fit a GAM model.
    
    Uses spline terms for continuous variables and factor terms for categoricals.
    """
    if categorical_encoded is None:
        categorical_encoded = []
    
    # Build GAM terms
    terms = None
    for i, fname in enumerate(feature_names):
        if fname in categorical_encoded:
            term = f(i)  # factor term for categorical
        else:
            term = s(i, n_splines=n_splines)  # spline term for continuous
        
        if terms is None:
            terms = term
        else:
            terms = terms + term
    
    # Fit GAM
    gam = LinearGAM(terms, lam=lam)
    gam.fit(X, y)
    
    print(f"\nGAM Summary:")
    print(f"  R² (pseudo): {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")
    print(f"  GCV score: {gam.statistics_['GCV']:.4f}")
    print(f"  AIC: {gam.statistics_['AIC']:.4f}")
    print(f"  Number of samples: {X.shape[0]}")
    print(f"  Number of features: {X.shape[1]}")
    
    return gam


def gridsearch_gam(X: np.ndarray, y: np.ndarray, feature_names: list,
                   categorical_encoded: list = None, n_splines: int = 20):
    """Fit GAM with automatic lambda selection via grid search."""
    if categorical_encoded is None:
        categorical_encoded = []
    
    terms = None
    for i, fname in enumerate(feature_names):
        if fname in categorical_encoded:
            term = f(i)
        else:
            term = s(i, n_splines=n_splines)
        
        if terms is None:
            terms = term
        else:
            terms = terms + term
    
    gam = GammaGAM(terms)
    
    # Grid search over lambda values
    lam_grid = np.logspace(-3, 3, 11)
    lam_combinations = [lam_grid] * len(feature_names)
    
    gam.gridsearch(X, y, lam=lam_combinations, progress=True)
    
    print(f"\nGAM (Grid Search) Summary:")
    print(f"  R² (pseudo): {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")
    print(f"  GCV score: {gam.statistics_['GCV']:.4f}")
    print(f"  AIC: {gam.statistics_['AIC']:.4f}")
    
    return gam


def plot_partial_dependence(gam, feature_names: list, X: np.ndarray, 
                            target_name: str, save_dir: str = None):
    """
    Plot partial dependence plots for each feature.
    These show the marginal effect of each variable on the target.
    """
    n_features = len(feature_names)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (ax, fname) in enumerate(zip(axes, feature_names)):
        XX = gam.generate_X_grid(term=i, meshgrid=False)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        
        ax.plot(XX[:, i], pdep, 'b-', linewidth=2)
        ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1], alpha=0.2, color='b')
        ax.set_xlabel(fname, fontsize=11)
        ax.set_ylabel(f'Partial effect on {target_name}', fontsize=10)
        ax.set_title(f'Effect of {fname}', fontsize=12)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Add rug plot showing data distribution
        ax.plot(X[:, i], [ax.get_ylim()[0]] * len(X[:, i]), '|', 
                color='gray', alpha=0.3, markersize=5)
    
    # Hide empty subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'GAM Partial Dependence Plots — Target: {target_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / f'gam_partial_dependence_{target_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_feature_importance(gam, feature_names: list, target_name: str,
                            save_dir: str = None):
    """
    Plot feature importance based on the chi-squared statistic 
    from each smooth term's significance test.
    """
    p_values = []
    chi2_stats = []
    
    for i in range(len(feature_names)):
        try:
            stat = gam.statistics_['p_values'][i]
            p_values.append(stat)
        except (KeyError, IndexError):
            p_values.append(1.0)
    
    # Use -log10(p_value) as importance measure
    importance = [-np.log10(max(p, 1e-300)) for p in p_values]
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    y_pos = range(len(feature_names))
    
    bars = ax.barh(y_pos, [importance[i] for i in sorted_idx], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('-log10(p-value)', fontsize=11)
    ax.set_title(f'Feature Significance — Target: {target_name}', fontsize=13)
    
    # Add significance threshold line
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', 
               label='p=0.05 threshold', alpha=0.7)
    ax.axvline(x=-np.log10(0.01), color='orange', linestyle='--', 
               label='p=0.01 threshold', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / f'gam_feature_importance_{target_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    
    # Print summary table
    print(f"\nFeature significance for target '{target_name}':")
    print(f"{'Feature':<30} {'p-value':<15} {'Significant (α=0.05)'}")
    print("-" * 65)
    for i in np.flip(sorted_idx):
        sig = "***" if p_values[i] < 0.001 else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else ""
        print(f"{feature_names[i]:<30} {p_values[i]:<15.6f} {sig}")


def evaluate_model(gam, X_train, y_train, X_test, y_test, target_name):
    """Evaluate GAM on train/test split."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_pred_train = gam.predict(X_train)
    y_pred_test = gam.predict(X_test)
    
    print(f"\nModel Evaluation — Target: {target_name}")
    print(f"{'Metric':<25} {'Train':<15} {'Test':<15}")
    print("-" * 55)
    print(f"{'R²':<25} {r2_score(y_train, y_pred_train):<15.4f} {r2_score(y_test, y_pred_test):<15.4f}")
    print(f"{'MAE':<25} {mean_absolute_error(y_train, y_pred_train):<15.4f} {mean_absolute_error(y_test, y_pred_test):<15.4f}")
    print(f"{'RMSE':<25} {np.sqrt(mean_squared_error(y_train, y_pred_train)):<15.4f} {np.sqrt(mean_squared_error(y_test, y_pred_test)):<15.4f}")
    
    return y_pred_test


def run_full_analysis(csv_path: str, target_col: str = None, 
                      feature_subset: list = None,
                      use_gridsearch: bool = False,
                      save_dir: str = None):
    """
    Run the complete GAM analysis pipeline.
    
    Args:
        csv_path: Path to eval_epoch_1.csv
        target_col: Which metric to model (e.g., 'ade', 'fde'). 
                    If None, will try to auto-detect.
        feature_subset: Optional list of specific feature columns to use.
        use_gridsearch: Whether to use grid search for lambda tuning.
        save_dir: Directory to save plots. If None, uses csv directory.
    """
    # Setup save directory
    if save_dir is None:
        save_dir = str(Path(csv_path).parent / 'gam_analysis')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)
    df = load_data(csv_path)
    
    # 2. Identify columns
    print("\n" + "=" * 70)
    print("STEP 2: Identifying Columns")
    print("=" * 70)
    feature_cols, categorical_cols, targets = identify_columns(df)
    
    # Allow manual override
    if feature_subset:
        feature_cols = [c for c in feature_subset if c in feature_cols]
        categorical_cols = [c for c in feature_subset if c in categorical_cols]
        print(f"\nUsing feature subset: {feature_cols + categorical_cols}")
    
    if target_col is None:
        if targets:
            target_col = targets[0]
            print(f"\nAuto-selected target: {target_col}")
        else:
            # If no standard target found, let user know available numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"\nNo standard target found. Available numeric columns:")
            for i, col in enumerate(numeric_cols):
                print(f"  {i}: {col}")
            raise ValueError(
                "Please specify target_col manually. "
                f"Available: {numeric_cols}"
            )
    
    assert target_col in df.columns, f"Target '{target_col}' not found in columns"
    
    # Remove target from features if present
    feature_cols = [c for c in feature_cols if c != target_col]
    
    if not feature_cols and not categorical_cols:
        raise ValueError("No feature columns identified! Please specify feature_subset.")
    
    # 3. Prepare data
    print("\n" + "=" * 70)
    print("STEP 3: Preparing Data")
    print("=" * 70)

    # Check correlations first
    corr, redundant = plot_correlation_matrix(df, feature_cols, save_dir)

    X, y, feature_names, cat_mappings, df_clean = prepare_data(
        df, feature_cols, categorical_cols, target_col
    )
    
    cat_encoded = [c + '_encoded' for c in categorical_cols]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4. Fit GAM
    print("\n" + "=" * 70)
    print("STEP 4: Fitting GAM")
    print("=" * 70)
    
    if use_gridsearch:
        gam = gridsearch_gam(X_train, y_train, feature_names, cat_encoded)
    else:
        gam = fit_gam(X_train, y_train, feature_names, cat_encoded)
    
    # 5. Evaluate
    print("\n" + "=" * 70)
    print("STEP 5: Model Evaluation")
    print("=" * 70)
    y_pred = evaluate_model(gam, X_train, y_train, X_test, y_test, target_col)
    
    # 6. Interpret
    print("\n" + "=" * 70)
    print("STEP 6: Interpretation Plots")
    print("=" * 70)
    plot_partial_dependence(gam, feature_names, X, target_col, save_dir)
    plot_feature_importance(gam, feature_names, target_col, save_dir)
    
    # 7. Print GAM summary
    print("\n" + "=" * 70)
    print("STEP 7: Full GAM Summary")
    print("=" * 70)
    gam.summary()
    
    return gam, feature_names, df_clean


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Path to your data
    CSV_PATH = (
        "experiments/trajectory_metrics_joined/"
        "nusc_mini_debug_tpp-06_Mar_2026_14_37_15/eval_epoch_1.csv"
    )

    # Features that describe the agent motion and scene context
    FEATURES = [
        'mean_speed', 'max_speed', 'std_speed',
        'mean_acceleration', 'max_acceleration',
        'mean_jerk', 'max_jerk',
        'path_efficiency', 'displacement', 'path_length',
        'heading_change', 'has_collision', 'min_neighbor_distance',
        'scene_num_agents', 'scene_num_VEHICLE',
        'scene_bbox_area', 'scene_bbox_width', 'scene_bbox_height',
        'scene_spatial_density', 'scene_density_VEHICLE',
    ]

    # Target: can be changed change to 'ml_fde' or 'min_ade_5'
    TARGET = 'ml_ade'

    gam, features, data = run_full_analysis(
        CSV_PATH,
        target_col=TARGET,
        feature_subset=FEATURES,
        use_gridsearch=False,  # Set True for better lambda tuning (slower)
    )