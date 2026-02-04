"""Run the F1 driver style clustering and prediction pipeline outside the notebook.

Usage examples:
    python run_model.py --db Databases/database.db --year 2025 --output model.joblib

The script collects features, fits a scaler->PCA->HDBSCAN pipeline, saves the model, and
prints a small cluster summary and driver-cluster assignment table.
"""
import argparse
import sqlite3
import sys
import os
import pandas as pd
import numpy as np

from f1_model.data import collect_race_data, DEFAULT_FEATURE_COLUMNS
from f1_model.model import F1DriverModel


def make_visuals(model: F1DriverModel, df: pd.DataFrame, outdir: str = 'visuals'):
    """Create several plots to highlight results and save them into outdir."""
    os.makedirs(outdir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        print("Matplotlib/seaborn not available; skipping visuals. Install: pip install matplotlib seaborn")
        return

    sns.set(style='whitegrid', context='talk')

    # Ensure PCA-transformed data available
    try:
        X = df[DEFAULT_FEATURE_COLUMNS].values
        Xs = model.scaler.transform(X)
        Xp = model.pca.transform(Xs)
    except Exception as e:
        print('Could not compute PCA transform for visuals:', e)
        return

    plot_df = df.copy()
    plot_df['PC1'] = Xp[:, 0] if Xp.shape[1] >= 1 else 0
    plot_df['PC2'] = Xp[:, 1] if Xp.shape[1] >= 2 else 0

    # PCA scatter colored by cluster; if only noise, color by driver (top drivers)
    unique_clusters = sorted(plot_df['cluster'].unique().tolist())
    only_noise = (len(unique_clusters) == 1 and unique_clusters[0] == -1)

    plt.figure(figsize=(12, 9))
    if only_noise:
        # color by driver: choose up to 8 most frequent drivers and group others
        top_drivers = plot_df['driver'].value_counts().nlargest(8).index.tolist()
        plot_df['driver_group'] = plot_df['driver'].where(plot_df['driver'].isin(top_drivers), 'OTHER')
        ax = sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='driver_group', palette='tab10', s=100, alpha=0.9)
        ax.set_title('PCA scatter (colored by driver) — HDBSCAN produced no clusters')
    else:
        plot_df['cluster_str'] = plot_df['cluster'].astype(str)
        ax = sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='cluster_str', palette='tab20', s=100, alpha=0.9)
        ax.set_title('PCA scatter (colored by cluster)')

    # annotate cluster centers if available
    if hasattr(model, '_driver_centroids') and model._driver_centroids is not None:
        cent = model._driver_centroids
        # cent columns named PC0..PCn
        cent_df = cent.reset_index()
        # map centroids to plotting PC names (PC0->PC1, PC1->PC2)
        if cent.shape[1] >= 2:
            xcol = cent.columns[0]
            ycol = cent.columns[1]
            for _, row in cent_df.iterrows():
                cx = row[xcol]
                cy = row[ycol]
                plt.scatter(cx, cy, marker='X', s=200, c='k')
                plt.text(cx, cy, row['driver'], fontsize=9, weight='bold')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    p1 = os.path.join(outdir, 'pca_scatter.png')
    plt.savefig(p1, dpi=200)
    plt.close()
    print('Saved PCA scatter to', p1)

    # Driver vs Cluster heatmap (percentage)
    try:
        clusterAssignment = pd.crosstab(df.driver, df.cluster)
        if not clusterAssignment.empty:
            clusterAssignmentPct = (clusterAssignment.div(clusterAssignment.sum(axis=0), axis=1) * 100).round(2)
            plt.figure(figsize=(14, max(4, clusterAssignmentPct.shape[0] * 0.3)))
            sns.heatmap(clusterAssignmentPct, cmap='rocket_r', annot=True, fmt='.1f')
            plt.title('Driver proportion per cluster (%)')
            plt.xlabel('Cluster')
            plt.ylabel('Driver')
            plt.tight_layout()
            p2 = os.path.join(outdir, 'driver_cluster_heatmap.png')
            plt.savefig(p2, dpi=200)
            plt.close()
            print('Saved driver-cluster heatmap to', p2)
    except Exception:
        pass

    # Optional UMAP/TSNE projection if available — useful for impressive visuals
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_proj = reducer.fit_transform(Xs)
        plot_df['UMAP1'] = umap_proj[:, 0]
        plot_df['UMAP2'] = umap_proj[:, 1]
        plt.figure(figsize=(12, 9))
        if only_noise:
            ax = sns.scatterplot(data=plot_df, x='UMAP1', y='UMAP2', hue='driver_group', palette='tab10', s=100, alpha=0.9)
            ax.set_title('UMAP projection (colored by driver)')
        else:
            ax = sns.scatterplot(data=plot_df, x='UMAP1', y='UMAP2', hue='cluster_str', palette='tab20', s=100, alpha=0.9)
            ax.set_title('UMAP projection (colored by cluster)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        p3 = os.path.join(outdir, 'umap_proj.png')
        plt.savefig(p3, dpi=200)
        plt.close()
        print('Saved UMAP projection to', p3)
    except Exception:
        # UMAP not installed — skip
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=r"Databases/database.db", help="Path to sqlite DB")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--output", default="f1_driver_model.joblib")
    p.add_argument("--averages", action="store_true", help="Aggregate per-driver per-race averages")
    p.add_argument("--visuals", action="store_true", help="Produce visuals into visuals/ folder")
    p.add_argument("--classifier", choices=['knn','rf'], default='rf', help="Supervised classifier to train in PCA space: 'rf' (RandomForest) or 'knn'")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute("SELECT raceID FROM Race WHERE year = ?", (args.year,))
    races = [r[0] for r in cur.fetchall()]
    if not races:
        print(f"No races found for year {args.year} in {args.db}")
        sys.exit(1)

    averages = args.averages or len(races) > 1
    df = collect_race_data(conn, races, DEFAULT_FEATURE_COLUMNS, averages=averages)
    if df.empty:
        print("No feature rows collected; exiting.")
        sys.exit(1)

    X = df[DEFAULT_FEATURE_COLUMNS].values
    model = F1DriverModel(DEFAULT_FEATURE_COLUMNS)

    try:
        # Train with supervised classifier (RandomForest) when driver labels are present for impressive results
        labels = model.fit(X, y=df['driver'].values, classifier=args.classifier)
    except RuntimeError as e:
        print("Runtime error while fitting model:", e)
        print("Make sure 'hdbscan' is installed in the environment: pip install hdbscan")
        sys.exit(1)

    df = df.reset_index(drop=True)
    df['cluster'] = labels

    clusterSummary = df.groupby('cluster')[DEFAULT_FEATURE_COLUMNS].mean()
    clusterAssignment = pd.crosstab(df.driver, df.cluster)
    clusterAssignmentPercentage = (clusterAssignment.div(clusterAssignment.sum(axis=0), axis=1) * 100).round(2)

    print("Cluster summary (first 10 rows):")
    print(clusterSummary.head(10))
    print('\nDriver -> cluster counts:')
    print(clusterAssignment.head(20))

    model._train_df = df
    model.save(args.output)
    print(f"Saved trained model to {args.output}")

    # save summaries
    clusterSummary.to_csv('cluster_summary.csv')
    clusterAssignment.to_csv('cluster_assignment.csv')
    clusterAssignmentPercentage.to_csv('cluster_assignment_percentage.csv')
    print('Wrote CSV summaries: cluster_summary.csv, cluster_assignment.csv, cluster_assignment_percentage.csv')

    if args.visuals:
        make_visuals(model, df, outdir='visuals')
        # additional supervised visuals when classifier available
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.model_selection import cross_val_predict
            from sklearn.metrics import confusion_matrix, classification_report
        except Exception:
            print('Matplotlib/seaborn or sklearn missing; skipping supervised visuals')
        else:
            Xs = model.scaler.transform(X)
            Xp = model.pca.transform(Xs)
            clf = getattr(model, '_clf', None)
            if clf is not None:
                classes = clf.classes_ if hasattr(clf, 'classes_') else None
                cv = min(5, max(2, len(np.unique(df['driver']))))
                try:
                    preds = cross_val_predict(clf, Xp, df['driver'].values, cv=cv)
                    cm = confusion_matrix(df['driver'].values, preds, labels=np.unique(df['driver']))
                    labels_list = np.unique(df['driver'])
                    plt.figure(figsize=(14, max(6, len(labels_list)*0.25)))
                    sns.heatmap(cm, xticklabels=labels_list, yticklabels=labels_list, cmap='Blues', annot=False)
                    plt.title(f'Confusion matrix (cv={cv})')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.tight_layout()
                    p4 = os.path.join('visuals', 'confusion_matrix.png')
                    plt.savefig(p4, dpi=200)
                    plt.close()
                    print('Saved confusion matrix to', p4)
                    # classification report
                    report = classification_report(df['driver'].values, preds, labels=labels_list, zero_division=0)
                    with open(os.path.join('visuals', 'classification_report.txt'), 'w') as fh:
                        fh.write(report)
                    print('Saved classification report to visuals/classification_report.txt')
                except Exception as e:
                    print('Could not compute cross-validated predictions for classifier:', e)

                # feature importances: if RF, map PC importances back to original features approx.
                if hasattr(clf, 'feature_importances_'):
                    pc_importances = clf.feature_importances_  # shape: n_pcs
                    # pca.components_ shape: n_pcs x n_features (components_ rows are PCs)
                    comps = model.pca.components_  # shape (n_pcs, n_features)
                    # approximate original feature importances as sum(|component_weight| * pc_importance)
                    orig_imp = np.sum(np.abs(comps.T) * pc_importances.reshape(1, -1), axis=1)
                    feat_imp = pd.Series(orig_imp, index=DEFAULT_FEATURE_COLUMNS)
                    feat_imp = feat_imp.sort_values(ascending=False)
                    plt.figure(figsize=(8, 4))
                    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='mako')
                    plt.title('Approximate original-feature importances (via PCA-weighted PC importances)')
                    plt.xlabel('Importance')
                    plt.tight_layout()
                    p5 = os.path.join('visuals', 'feature_importances.png')
                    plt.savefig(p5, dpi=200)
                    plt.close()
                    print('Saved feature importances to', p5)
                    # save numeric importances to CSV
                    feat_imp.to_csv(os.path.join('visuals', 'feature_importances.csv'))
            else:
                print('No supervised classifier available; skipped confusion matrix and feature importances')


if __name__ == '__main__':
    main()
