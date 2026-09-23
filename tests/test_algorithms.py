"""
Comprehensive unit and sanity test suite for all from-scratch ML algorithms in heart-disease-ml.
Verifies convergence, shapes, invariants, and correctness on synthetic datasets.
No scikit-learn dependencies.
"""

import numpy as np
import pandas as pd
import unittest

from src.data_utils import train_test_split
from src.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    calculate_metrics,
)
from src.preprocessing import HeartDiseasePreprocessor
from src.models.perceptron import PerceptronClassifier
from src.models.logistic_regression import LogisticRegression
from src.models.knn_classifier import KNNClassifier
from src.models.neural_network import MLPClassifier
from src.dimensionality_reduction.pca import PCA
from src.dimensionality_reduction.svd import TruncatedSVD
from src.dimensionality_reduction.lda import LDA
from src.models.svm_hard_margin import HardMarginSVM
from src.models.svm_soft_margin import SoftMarginSVM
from src.models.kernel_svm import KernelSVM
from src.models.multiclass_svm import OneVsRestSVM, OneVsOneSVM


class TestHeartDiseaseML(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_01_data_utils_train_test_split(self):
        """Test stratified splitting and ratio preservation."""
        X = np.random.randn(100, 4)
        y = np.array([0] * 60 + [1] * 40)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)
        # Check ratio: 60/40 => in 20 test, expect 12 zeros and 8 ones
        self.assertEqual(np.sum(y_te == 0), 12)
        self.assertEqual(np.sum(y_te == 1), 8)

        # DataFrame / Series test
        df_X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        s_y = pd.Series(y, name="target")
        X_tr_df, X_te_df, y_tr_s, y_te_s = train_test_split(df_X, s_y, test_size=0.2, random_state=42, stratify=s_y)
        self.assertIsInstance(X_tr_df, pd.DataFrame)
        self.assertIsInstance(y_te_s, pd.Series)
        self.assertEqual(len(X_tr_df), 80)
        self.assertEqual(np.sum(y_te_s == 0), 12)
        self.assertEqual(np.sum(y_te_s == 1), 8)

    def test_02_metrics(self):
        """Test all evaluation metrics against exact known values."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        # TP=3, FN=1, TN=3, FP=1
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 6 / 8)
        self.assertAlmostEqual(precision_score(y_true, y_pred), 3 / 4)
        self.assertAlmostEqual(recall_score(y_true, y_pred), 3 / 4)
        self.assertAlmostEqual(f1_score(y_true, y_pred), 3 / 4)

        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.tolist(), [[3, 1], [1, 3]])

        # Perfect prediction AUC
        y_score = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.5, 0.1])
        auc = roc_auc_score(y_true, y_score)
        self.assertGreaterEqual(auc, 0.8)

        # Dictionary format
        m_dict = calculate_metrics(y_true, y_pred, y_score)
        self.assertIn("Accuracy", m_dict)
        self.assertIn("ROC_AUC", m_dict)

    def test_03_preprocessing(self):
        """Test HeartDiseasePreprocessor for leakage prevention and encoding."""
        data = {
            "age": [50.0, np.nan, 60.0, 45.0],
            "trestbps": [120.0, 140.0, np.nan, 130.0],
            "chol": [200.0, 240.0, 220.0, np.nan],
            "thalach": [150.0, 160.0, 140.0, 170.0],
            "oldpeak": [1.0, 2.0, 0.0, 1.5],
            "sex": [1, 0, 1, 0],
            "cp": [0, 1, 2, 0],
            "fbs": [0, 1, 0, 0],
            "restecg": [0, 2, 1, 0],
            "exang": [0, 1, 0, 1],
            "slope": [1, 2, 1, 2],
            "ca": [0, 1, 2, 0],
            "thal": [0, 2, 1, 0],
        }
        df = pd.DataFrame(data)
        prep = HeartDiseasePreprocessor()
        X_proc = prep.fit_transform(df)
        self.assertFalse(np.isnan(X_proc).any())
        self.assertEqual(X_proc.dtype, np.float64)

        # Test unseen category maps to 0
        df_unseen = df.copy()
        df_unseen.loc[0, "cp"] = 99
        X_unseen = prep.transform(df_unseen)
        self.assertFalse(np.isnan(X_unseen).any())

    def test_04_perceptron_convergence(self):
        """Test Perceptron convergence on linearly separable dataset."""
        # 2D linearly separable clusters
        X0 = np.random.randn(25, 2) - 3.0
        X1 = np.random.randn(25, 2) + 3.0
        X = np.vstack([X0, X1])
        y = np.array([0] * 25 + [1] * 25)

        clf = PerceptronClassifier(learning_rate=0.05, epochs=200, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        self.assertEqual(acc, 1.0)

    def test_05_logistic_regression_loss_decrease(self):
        """Test Logistic Regression loss decreases and separates clusters."""
        X0 = np.random.randn(30, 2) - 2.0
        X1 = np.random.randn(30, 2) + 2.0
        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)

        lr = LogisticRegression(learning_rate=0.05, epochs=500, random_state=42)
        lr.fit(X, y)
        self.assertLess(lr.loss_history_[-1], lr.loss_history_[0])
        acc = accuracy_score(y, lr.predict(X))
        self.assertGreaterEqual(acc, 0.95)

    def test_06_knn_classification(self):
        """Test KNN majority voting and distance weighting."""
        X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        knn_u = KNNClassifier(n_neighbors=3, weights="uniform").fit(X, y)
        self.assertEqual(knn_u.predict(np.array([[-2.5]]))[0], 0)
        self.assertEqual(knn_u.predict(np.array([[2.5]]))[0], 1)

        knn_d = KNNClassifier(n_neighbors=3, weights="distance").fit(X, y)
        self.assertEqual(knn_d.predict(np.array([[-2.5]]))[0], 0)
        self.assertEqual(knn_d.predict(np.array([[2.5]]))[0], 1)

    def test_07_mlp_classifier(self):
        """Test MLPClassifier training and loss reduction."""
        X0 = np.random.randn(40, 4) - 1.5
        X1 = np.random.randn(40, 4) + 1.5
        X = np.vstack([X0, X1])
        y = np.array([0] * 40 + [1] * 40)

        mlp = MLPClassifier(hidden_layers=(16, 8), epochs=150, learning_rate=0.02, random_state=42)
        mlp.fit(X, y)
        self.assertLess(mlp.loss_history_[-1], mlp.loss_history_[0])
        acc = accuracy_score(y, mlp.predict(X))
        self.assertGreaterEqual(acc, 0.90)

    def test_08_pca(self):
        """Test PCA projection, components, and explained variance."""
        X = np.random.randn(50, 6)
        pca = PCA(n_components=3).fit(X)
        X_trans = pca.transform(X)
        self.assertEqual(X_trans.shape, (50, 3))
        self.assertEqual(pca.components_.shape, (3, 6))
        self.assertEqual(len(pca.explained_variance_ratio_), 3)
        self.assertAlmostEqual(float(np.sum(pca.explained_variance_ratio_)), float(np.sum(pca.explained_variance_) / np.sum(np.var(X, axis=0, ddof=1))))

        # Variance ratio threshold
        pca_ratio = PCA(n_components=0.90).fit(X)
        self.assertGreaterEqual(np.sum(pca_ratio.explained_variance_ratio_), 0.90)

    def test_09_truncated_svd(self):
        """Test SVD shapes and projection."""
        X = np.random.randn(40, 5)
        svd = TruncatedSVD(n_components=2).fit(X)
        X_proj = svd.transform(X)
        self.assertEqual(X_proj.shape, (40, 2))
        self.assertEqual(svd.components_.shape, (2, 5))
        self.assertEqual(len(svd.singular_values_), 2)

    def test_10_lda(self):
        """Test supervised LDA projection on binary dataset."""
        X0 = np.random.randn(30, 4) - 2.0
        X1 = np.random.randn(30, 4) + 2.0
        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)

        lda = LDA(n_components=1).fit(X, y)
        X_proj = lda.transform(X)
        self.assertEqual(X_proj.shape, (60, 1))
        # Check that projected means of class 0 and 1 are well separated
        mean0 = np.mean(X_proj[:30])
        mean1 = np.mean(X_proj[30:])
        self.assertNotAlmostEqual(mean0, mean1, places=1)

    def test_11_hard_margin_svm(self):
        """Test Hard-Margin SVM on separable toy dataset."""
        X0 = np.array([[-3.0, -3.0], [-2.0, -2.0], [-2.5, -1.5]])
        X1 = np.array([[3.0, 3.0], [2.0, 2.0], [2.5, 1.5]])
        X = np.vstack([X0, X1])
        y = np.array([0, 0, 0, 1, 1, 1])

        svm = HardMarginSVM(epochs=500, learning_rate=0.01).fit(X, y)
        preds = svm.predict(X)
        self.assertEqual(preds.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertTrue(svm.is_separable_)

    def test_12_soft_margin_svm(self):
        """Test Soft-Margin SVM with outliers."""
        X0 = np.array([[-2.0, -2.0], [-1.5, -1.5], [-1.0, -1.0]])
        X1 = np.array([[2.0, 2.0], [1.5, 1.5], [1.0, 1.0]])
        # Add slight overlap / outlier
        X_outlier = np.array([[-0.5, -0.5], [0.5, 0.5]])
        X = np.vstack([X0, X1, X_outlier])
        y = np.array([0, 0, 0, 1, 1, 1, 1, 0])

        svm = SoftMarginSVM(C=1.0, epochs=1000, learning_rate=0.005).fit(X, y)
        scores = svm.decision_function(X)
        self.assertEqual(len(scores), 8)
        self.assertIsNotNone(svm.support_vectors_)

    def test_13_kernel_svm_nonlinear_xor(self):
        """Test Kernel SVM on XOR nonlinear dataset."""
        # XOR dataset: (1, 1) and (-1, -1) -> 0; (1, -1) and (-1, 1) -> 1
        X = np.array([
            [1.0, 1.0], [1.2, 0.8], [-1.0, -1.0], [-0.8, -1.2],
            [1.0, -1.0], [0.8, -1.2], [-1.0, 1.0], [-1.2, 0.8]
        ])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Linear SVM should fail on XOR
        linear_svm = KernelSVM(kernel="linear", epochs=100).fit(X, y)
        linear_acc = accuracy_score(y, linear_svm.predict(X))

        # RBF Kernel SVM should succeed on XOR
        rbf_svm = KernelSVM(kernel="rbf", gamma=1.0, C=10.0, epochs=300).fit(X, y)
        rbf_acc = accuracy_score(y, rbf_svm.predict(X))
        self.assertEqual(rbf_acc, 1.0)
        self.assertGreater(rbf_acc, linear_acc)

    def test_14_multiclass_svm(self):
        """Test OneVsRestSVM and OneVsOneSVM on 3-class dataset."""
        X0 = np.random.randn(15, 2) + np.array([-4.0, -4.0])
        X1 = np.random.randn(15, 2) + np.array([0.0, 4.0])
        X2 = np.random.randn(15, 2) + np.array([4.0, -4.0])
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * 15 + [1] * 15 + [2] * 15)

        ovr = OneVsRestSVM(estimator_cls=KernelSVM, kernel="rbf", C=5.0, epochs=150).fit(X, y)
        preds = ovr.predict(X)
        acc = accuracy_score(y, preds)
        self.assertGreaterEqual(acc, 0.90)

        ovo = OneVsOneSVM(estimator_cls=KernelSVM, kernel="rbf", C=5.0, epochs=150).fit(X, y)
        ovo_preds = ovo.predict(X)
        ovo_acc = accuracy_score(y, ovo_preds)
        self.assertGreaterEqual(ovo_acc, 0.90)


if __name__ == "__main__":
    unittest.main()
