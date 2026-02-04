import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from typing import List, Optional

try:
    from hdbscan import HDBSCAN
    from hdbscan import approximate_predict
except Exception:
    HDBSCAN = None
    approximate_predict = None

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score


class F1DriverModel:
    """Encapsulates scaler -> PCA -> HDBSCAN pipeline and helper methods."""

    def __init__(self, feature_columns: List[str], n_components: Optional[float] = 0.95,
                 hdbscan_kwargs: Optional[dict] = None):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.hdbscan_kwargs = hdbscan_kwargs or {
            "min_cluster_size": 8,
            "min_samples": 2,
            "cluster_selection_method": "leaf",
            "metric": "euclidean",
            "prediction_data": True,
        }
        self.hdb = None
        self._clf = None

    def fit(self, X: np.ndarray, y: Optional[list] = None, classifier: str = 'knn'):
        """Fit scaler, PCA, HDBSCAN. If y (driver labels) provided, also fit a supervised classifier in PCA space.

        Returns: labels from HDBSCAN (array-like)
        """
        X = np.asarray(X, dtype=float)
        Xs = self.scaler.fit_transform(X)
        Xp = self.pca.fit_transform(Xs)
        if HDBSCAN is None:
            raise RuntimeError("hdbscan is not installed in the environment")
        self.hdb = HDBSCAN(**self.hdbscan_kwargs)
        labels = self.hdb.fit_predict(Xp)

        # supervised fallback: train a classifier in PCA space if labels provided
        if y is not None:
            try:
                y_arr = np.asarray(y)
                classes = np.unique(y_arr)
                if classifier == 'rf':
                    # RandomForest on PCA components (small trees for demo)
                    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                    rf.fit(Xp, y_arr)
                    self._clf = rf
                    # store a quick CV accuracy for diagnostics
                    try:
                        preds = cross_val_predict(rf, Xp, y_arr, cv=min(5, max(2, len(classes))))
                        self._clf_cv_accuracy = float(accuracy_score(y_arr, preds))
                    except Exception:
                        self._clf_cv_accuracy = None
                else:
                    knn = KNeighborsClassifier(n_neighbors=min(3, max(1, len(classes))))
                    knn.fit(Xp, y_arr)
                    self._clf = knn
                    self._clf_cv_accuracy = None
            except Exception:
                self._clf = None
                self._clf_cv_accuracy = None

        return labels

    def predict(self, X: np.ndarray):
        if self.hdb is None:
            raise RuntimeError("Model not fitted")
        Xs = self.scaler.transform(X)
        Xp = self.pca.transform(Xs)
        if approximate_predict is None:
            raise RuntimeError("hdbscan approximate_predict not available")
        labels, probs = approximate_predict(self.hdb, Xp)
        return labels, probs

    def predict_with_fallback(self, X: np.ndarray):
        """Predict cluster via HDBSCAN; if cluster is noise (-1) or HDBSCAN not available, use classifier probabilities if present."""
        labels, probs = self.predict(X)
        # if label is noise and classifier present, use classifier
        if (labels[0] == -1 or probs[0] == 0.0) and self._clf is not None:
            # compute classifier probs in PCA space
            Xs = self.scaler.transform(X)
            Xp = self.pca.transform(Xs)
            if hasattr(self._clf, 'predict_proba'):
                clf_probs = self._clf.predict_proba(Xp)[0]
                classes = self._clf.classes_
                # return classes and probs as mapped
                return classes, clf_probs
        return labels, probs

    def predict_supervised_proba(self, X: np.ndarray):
        """Return supervised classifier classes and probabilities if trained, else raise."""
        if self._clf is None:
            raise RuntimeError('No supervised classifier is available')
        Xs = self.scaler.transform(X)
        Xp = self.pca.transform(Xs)
        if not hasattr(self._clf, 'predict_proba'):
            raise RuntimeError('Classifier does not support predict_proba')
        probs = self._clf.predict_proba(Xp)
        return self._clf.classes_, probs

    def save(self, path: str):
        joblib.dump({
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
            "pca": self.pca,
            "hdbscan": self.hdb,
            # optionally save training dataframe if attached
            "train_df": getattr(self, "_train_df", None),
            "clf": getattr(self, "_clf", None),
            "clf_cv_accuracy": getattr(self, "_clf_cv_accuracy", None),
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        inst = cls(data["feature_columns"])  # we'll restore fitted objects
        inst.scaler = data.get("scaler")
        inst.pca = data.get("pca")
        inst.hdb = data.get("hdbscan")
        train_df = data.get("train_df")
        if train_df is not None:
            inst._train_df = train_df
        inst._clf = data.get("clf")
        inst._clf_cv_accuracy = data.get("clf_cv_accuracy")
        return inst
