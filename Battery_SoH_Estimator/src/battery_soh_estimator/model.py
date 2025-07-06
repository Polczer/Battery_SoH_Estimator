import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from battery_soh_estimator.features.core import features_func
from battery_soh_estimator.features.prod.features_prod import prod_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


class BatterySoHModel:
    def __init__(
        self,
        model_params: dict = None,
        selected_features: list = None,
        use_cv: bool = False,
        cv_folds: int = 5,
        aggregation_params: dict = None,
    ):
        """
        Initialize the BatterySoHModel with enhanced feature processing.

        Args:
            model_params: Parameters for the RandomForestRegressor
            selected_features: List of feature names to use from prod_features
            use_cv: Whether to perform cross-validation during training
            cv_folds: Number of folds for cross-validation
            aggregation_params: Parameters for feature aggregation
        """
        self.model_params = model_params or {
            "n_estimators": 200,
            "random_state": 35,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "n_jobs": -1,
        }
        self._init_selected_features = selected_features or list(prod_features.keys())
        self._update_processing_functions()
        self._actual_features = None
        self._agg_features = None
        self._min_soh = None
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.is_trained = False

        # Aggregation configuration
        self.aggregation_params = aggregation_params or {
            "numeric_agg_funcs": ["mean", "std", "min", "max"],
            "categorical_features": "nearest_temp",
            "weight_features": ["weight_0", "weight_25", "weight_45"],
        }

        self.model = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """Construct the scikit-learn pipeline."""
        return Pipeline(
            [
                ("regressor", RandomForestRegressor(**self.model_params)),
            ]
        )

    def _update_processing_functions(self):
        """Update the processing functions based on selected features."""
        self.processing_functions = {
            name: func
            for name, func in prod_features.items()
            if self._init_selected_features is None
            or name in self._init_selected_features
        }

    def _apply_monotonic_constraint(self, soh_pred: np.ndarray) -> np.ndarray:
        """Apply monotonic decreasing constraint to predictions."""
        if self._min_soh is None:
            self._min_soh = np.min(soh_pred)
        else:
            soh_pred = np.minimum(soh_pred, self._min_soh)
            self._min_soh = np.min(soh_pred)
        return soh_pred

    def _check_required_columns(
        self, df: pd.DataFrame, required_cols: list[str]
    ) -> None:
        """
        Check that all required columns are present in the DataFrame.
        """
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _get_numeric_features(self, df: pd.DataFrame, exclude: list[str]) -> list[str]:
        """
        Return numeric feature columns excluding specified ones.
        """
        return [
            col
            for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]

    def _aggregate_numeric(
        self, df: pd.DataFrame, features: list[str], funcs: list[str]
    ) -> dict:
        """
        Apply aggregation functions to numeric features.
        """
        agg = {}
        for col in features:
            for func in funcs:
                try:
                    agg[f"{col}_{func}"] = df[col].agg(func)
                except Exception as e:
                    raise ValueError(f"Failed to aggregate {col} with {func}: {str(e)}")
        return agg

    def _aggregate_weights(self, df: pd.DataFrame, weights: list[str]) -> dict:
        """
        Compute mean of weight features.
        """
        return {col: df[col].mean() for col in weights if col in df.columns}

    def _aggregate_categorical(self, df: pd.DataFrame, cat_feature: str) -> dict:
        """
        Compute mode of a categorical feature.
        """
        try:
            mode_val = df[cat_feature].mode()
            value = mode_val.iloc[0] if not mode_val.empty else np.nan
        except Exception as e:
            raise ValueError(f"Failed to get mode for {cat_feature}: {str(e)}")
        return {f"{cat_feature}_mode": value}

    def _aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates features from a single data chunk into a single row.

        Handles numeric, weight, and categorical features with proper error checking.

        Args:
            df: Input DataFrame for a single chunk

        Returns:
            Single-row DataFrame with aggregated features

        Raises:
            ValueError: If required columns are missing or aggregation fails
        """
        if df.empty:
            raise ValueError("Cannot aggregate empty DataFrame")

        # Get aggregation parameters
        agg_funcs = self.aggregation_params["numeric_agg_funcs"]
        cat_feature = self.aggregation_params["categorical_features"]
        weights = self.aggregation_params["weight_features"]

        # Check that all required columns are present
        self._check_required_columns(df, weights + [cat_feature])

        # Determine numeric columns to aggregate (excluding weight, categorical and target)
        numeric_features = self._get_numeric_features(
            df, exclude=weights + [cat_feature, "SoH"]
        )

        # Aggregate different types of features
        aggregated = {}
        aggregated.update(self._aggregate_numeric(df, numeric_features, agg_funcs))
        aggregated.update(self._aggregate_weights(df, weights))
        aggregated.update(self._aggregate_categorical(df, cat_feature))

        # Return result as single-row DataFrame
        try:
            return pd.DataFrame(aggregated, index=[0])
        except Exception as e:
            raise ValueError(f"Failed to create result DataFrame: {str(e)}")

    def extract_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from raw battery data."""
        return features_func.extract_features(df, self.processing_functions)

    def fit(
        self,
        df_or_X: pd.DataFrame,
        y: np.ndarray,
        features_already_extracted: bool = False,
    ) -> None:
        """
        Train the model on pre-aggregated or pre-extracted features.

        Args:
            df_or_X: DataFrame with raw data or features
            y: Target SoH values
            features_already_extracted: Set True if df_or_X already contains extracted features
        """
        if features_already_extracted:
            X = df_or_X
        else:
            df_feat = self.extract_X(df_or_X)
            self._actual_features = df_feat.columns.tolist()
            X = self._aggregate_features(df_feat)
            self._agg_features = X.columns.tolist()

        if len(X) != len(y):
            raise ValueError(
                f"Length mismatch: X has {len(X)} rows, y has {len(y)}"
            )

        if self.use_cv:
            print(f"[INFO] Performing {self.cv_folds}-fold cross-validation...")
            scores = cross_val_score(
                self.model, X, y, cv=self.cv_folds, scoring="neg_mean_absolute_error"
            )
            print(f"[INFO] CV MAE: {-np.mean(scores):.4f} ± {np.std(scores):.4f}")

        self.model.fit(X, y)
        self.is_trained = True

    def predict(
        self,
        df_or_X: pd.DataFrame,
        features_already_extracted: bool = False,
        aggregate: bool = True,
        chunk_size: int = None,
        monotonic: bool = True,
        reset_state: bool = True,
    ) -> np.ndarray:
        """
        Predict SoH with memory-efficient chunked processing if specified.

        Args:
            df_or_X: Input data (either raw battery DataFrame or pre-extracted features)
            features_already_extracted: Set to True if df_or_X already contains extracted features
            aggregate: Whether to aggregate features before prediction
            chunk_size: Optional size of chunks for streaming prediction
            monotonic: Whether to apply monotonic smoothing (non-increasing SoH)

        Returns:
            np.ndarray: Predicted SoH values

        Raises:
            RuntimeError: If the model is not trained
            ValueError: If no valid chunks are found or input is invalid

        Example:
        >>> model = BatterySoHModel()
        >>> model.fit(train_df, train_soh)
        >>> soh_preds = model.predict(test_df)
        >>> soh_preds.shape
        (1,)

        >>> # For chunked prediction
        >>> soh_preds = model.predict(test_df, chunk_size=100)
        >>> soh_preds[:5]
        array([0.985, 0.982, 0.980, 0.976, 0.974])
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        if reset_state:
            self._min_soh = None  # First call reset

        # For prepared data
        if features_already_extracted:
            X = df_or_X
            if aggregate:
                self._agg_features = X.columns.tolist()
            else:
                self._actual_features = X.columns.tolist()
            soh_pred = self.model.predict(X)
            return self._apply_monotonic_constraint(soh_pred) if monotonic else soh_pred

        # Streaming prediction
        if chunk_size is not None:
            preds = []
            for start in range(0, len(df_or_X), chunk_size):
                chunk = df_or_X.iloc[start: start + chunk_size]
                if len(chunk) < chunk_size // 2:
                    continue

                try:
                    chunk_features = self.extract_X(chunk)
                    agg_features = self._aggregate_features(chunk_features) if aggregate else chunk_features
                    soh_pred = self.model.predict(agg_features)[0]  # One chunk prediction

                    if monotonic:
                        soh_pred = self._apply_monotonic_constraint(np.array([soh_pred]))[0]

                    preds.append(soh_pred)
                except Exception as e:
                    print(f"[Warning] Skipped chunk at {start}: {e}")
                    continue

            if not preds:
                raise ValueError("No valid chunks found for aggregation.")

            return np.array(preds)

        # No chunks
        df_feat = self.extract_X(df_or_X)
        self._actual_features = df_feat.columns.tolist()
        X = self._aggregate_features(df_feat) if aggregate else df_feat
        self._agg_features = X.columns.tolist() if aggregate else []
        soh_pred = self.model.predict(X)
        return self._apply_monotonic_constraint(soh_pred) if monotonic else soh_pred

    def reset_state(self) -> None:
        """Reset the monotonic constraint state."""
        self._min_soh = None

    def save(self, path="battery_model.pkl"):
        """Save the model and its configuration to a file."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        save_data = {
            "model": self.model,
            "init_selected_features": self._init_selected_features,
            "actual_features": self._actual_features,
            "agg_features": self._agg_features,
            "use_cv": self.use_cv,
            "cv_folds": self.cv_folds,
            "model_params": self.model_params,
            "aggregation_params": self.aggregation_params,
        }
        try:
            joblib.dump(save_data, path)
            print(f"[INFO] Model saved to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {str(e)}")

    def load(self, path="battery_model.pkl"):
        """Load the model and its configuration from a file."""
        data = joblib.load(path)
        self.model = data["model"]
        self._init_selected_features = data["init_selected_features"]
        self.processing_functions = {
            name: func
            for name, func in prod_features.items()
            if name in self._init_selected_features
        }
        self.best_params_ = data.get("best_params", None)
        self.use_cv = data.get("use_cv", False)
        self.cv_folds = data.get("cv_folds", 5)
        self.is_trained = True
        print(f"[INFO] Model loaded from {path}")

    def plot_feature_importance(self):
        """
        Plot feature importance from the trained model.

        Only applicable for models that expose the `feature_importances_` attribute.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before plotting feature importance."
            )

        regressor = self.model.named_steps["regressor"]

        if not hasattr(regressor, "feature_importances_"):
            raise AttributeError("Model does not support feature importance")

        importances = regressor.feature_importances_
        feature_names = self._agg_features or self._actual_features

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances)
        plt.xlabel("Feature Importance")
        plt.title("Feature Importance in Random Forest")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @property
    def selected_features(self) -> list:
        """Get the currently selected features."""
        return self._init_selected_features

    @property
    def actual_features(self) -> list:
        """Get the features actually used in training/prediction."""
        return self._actual_features or []

    @property
    def aggregated_features(self) -> list:
        """Get the aggregated features actually used in training/prediction."""
        return self._agg_features or []