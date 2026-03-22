"""
Neural Regime Classifier (PyTorch LSTM)
=======================================
Independent ML model for economic regime classification.
Uses the same 24 base features as XGBoost plus experimental features
(e.g. oil_yoy).  Does NOT affect the existing HMM+GMM+XGBoost ensemble.

Architecture:
  LayerNorm → LSTM (1 layer) → Temporal Attention → Linear → 5-class softmax

Designed for small datasets (~800 monthly observations):
  - Small hidden size (32) to prevent overfitting
  - Long sequence length (48 months) to capture full business cycles
  - Focal loss for extreme class imbalance
  - Temporal oversampling of minority classes
  - Early stopping with patience
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from config.settings import MODEL_DIR, N_REGIMES, REGIME_LABELS

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
#  PyTorch Model Components
# =====================================================================


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification with extreme class imbalance.
    Down-weights well-classified (easy) examples so the model focuses on
    hard/rare classes like Crisis (21 samples) vs Expansion (641 samples).
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha          # class weights tensor
        self.gamma = gamma          # focusing parameter
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        # Apply label smoothing: correct class gets (1 - smoothing), others get smoothing/(n-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Per-class focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Combine: alpha * focal_weight * smooth_cross_entropy
        loss = -self.alpha.unsqueeze(0) * focal_weight * smooth_targets * log_probs
        return loss.sum(dim=-1).mean()


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM time steps."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) — attention-weighted sum
            weights: (batch, seq_len) — attention weights
        """
        scores = self.attn(lstm_output).squeeze(-1)       # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)            # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden)
        return context, weights


class LSTMRegimeModel(nn.Module):
    """
    Unidirectional LSTM with temporal attention for 5-class regime classification.
    Deliberately small (~8K params) to match dataset size (~830 sequences).
    """

    def __init__(self, n_features: int, n_classes: int = N_REGIMES,
                 hidden_size: int = 32, n_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.layer_norm = nn.LayerNorm(n_features)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )

        lstm_out_size = hidden_size  # unidirectional
        self.attention = TemporalAttention(lstm_out_size)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, n_classes)
            attn_weights: (batch, seq_len)
        """
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)                    # (batch, seq_len, hidden)
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden)
        logits = self.classifier(context)             # (batch, n_classes)
        return logits, attn_weights


# =====================================================================
#  Classifier Wrapper (mirrors EnsembleRegimeClassifier interface)
# =====================================================================


class NeuralRegimeClassifier:
    """
    PyTorch LSTM regime classifier.  Independent from the HMM+GMM+XGBoost ensemble.
    Accepts the same feature matrix + optional extra features (e.g. oil_yoy).
    """

    def __init__(self, n_regimes: int = N_REGIMES, seq_len: int = 24,
                 hidden_size: int = 32, random_state: int = 42):
        self.n_regimes = n_regimes
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.random_state = random_state

        self.model: Optional[LSTMRegimeModel] = None
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.is_fitted = False
        self._col_medians: Optional[pd.Series] = None
        self._label_params: Optional[Dict] = None
        self._training_metadata: Dict = {}
        self._last_attn_weights: Optional[np.ndarray] = None
        self._last_input_tensor: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------
    #  Training
    # -----------------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        extra_features: Optional[pd.DataFrame] = None,
    ) -> "NeuralRegimeClassifier":
        """
        Train the LSTM classifier.

        Args:
            features: Base feature matrix (24 cols + 'recession' column)
                      from build_feature_matrix(fred_df, shiller_df).
                      Includes cape/cape_zscore from Shiller data.
            extra_features: Optional DataFrame with experimental features
                           (e.g. oil_yoy). Same DatetimeIndex as features.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        logger.info("Training Neural Regime Classifier (LSTM)...")

        if "recession" not in features.columns:
            raise ValueError("Feature matrix must contain 'recession' column")

        target = features["recession"].copy()
        X_raw = features.drop(columns=["recession"]).copy()

        # Merge experimental features
        if extra_features is not None and not extra_features.empty:
            common_idx = X_raw.index.intersection(extra_features.index)
            extra_aligned = extra_features.reindex(X_raw.index)
            X_raw = pd.concat([X_raw, extra_aligned], axis=1)
            logger.info(
                f"  Merged {len(extra_features.columns)} experimental features: "
                f"{list(extra_features.columns)}"
            )

        self.feature_cols = list(X_raw.columns)
        n_features = len(self.feature_cols)
        logger.info(f"  Total features: {n_features} ({n_features - 24} experimental)")

        # NaN handling (same strategy as EnsembleRegimeClassifier)
        X_raw = X_raw.ffill().bfill()
        n_feature_cols = X_raw.shape[1]
        valid_mask = (
            (X_raw.notna().sum(axis=1) >= int(n_feature_cols * 0.6))
            & target.notna()
        )
        X_raw = X_raw[valid_mask]
        target = target[valid_mask]

        col_medians = X_raw.median()
        still_nan = col_medians.isna()
        if still_nan.any():
            nan_cols = list(col_medians[still_nan].index)
            logger.info(f"  Columns with no data (filled 0): {nan_cols}")
            col_medians = col_medians.fillna(0.0)
        X_raw = X_raw.fillna(col_medians)
        self._col_medians = col_medians

        if X_raw.isna().sum().sum() > 0:
            X_raw = X_raw.fillna(0.0)

        # Replace inf/-inf with NaN, then fill with 0
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        logger.info(f"  Training data: {len(X_raw)} rows, {n_features} features")

        # Scale
        X_scaled = self.scaler.fit_transform(X_raw)
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Create 5-class labels (reuse logic from EnsembleRegimeClassifier)
        labels = self._create_labels(X_raw, target)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, labels.values)
        logger.info(f"  Sequences: {len(X_seq)} (seq_len={self.seq_len})")

        if len(X_seq) < 50:
            raise ValueError(f"Too few sequences ({len(X_seq)}). Need at least 50.")

        # Train/validation split (last 20%, time-ordered)
        val_size = max(20, int(len(X_seq) * 0.20))
        X_train, X_val = X_seq[:-val_size], X_seq[-val_size:]
        y_train, y_val = y_seq[:-val_size], y_seq[-val_size:]

        # Moderate oversampling of minority classes (2x max)
        X_train, y_train = self._oversample_minority(X_train, y_train)

        # Log class distributions
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        val_classes_arr, val_counts = np.unique(y_val, return_counts=True)
        logger.info(f"  Train ({len(y_train)}): {dict(zip([REGIME_LABELS.get(c,'?') for c in train_classes], train_counts))}")
        logger.info(f"  Val ({len(y_val)}):   {dict(zip([REGIME_LABELS.get(c,'?') for c in val_classes_arr], val_counts))}")

        # Class weights — use sqrt scaling to avoid extreme weights
        # Raw inverse-frequency gives Crisis ~26x Expansion weight, which is too aggressive
        # sqrt scaling gives ~5x which is a better balance
        classes, counts = np.unique(y_train, return_counts=True)
        raw_weights = {c: len(y_train) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        class_weights = torch.zeros(self.n_regimes, device=DEVICE)
        for c, w in raw_weights.items():
            if c < self.n_regimes:
                class_weights[int(c)] = np.sqrt(w)  # sqrt dampening
        # Fill any missing classes with 1.0
        class_weights[class_weights == 0] = 1.0
        # Normalize so mean weight = 1.0
        class_weights = class_weights / class_weights.mean()
        logger.info(f"  Class weights: {dict(zip([REGIME_LABELS.get(int(c),'?') for c in classes], class_weights.cpu().tolist()))}")

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(DEVICE)
        y_train_t = torch.LongTensor(y_train).to(DEVICE)
        X_val_t = torch.FloatTensor(X_val).to(DEVICE)
        y_val_t = torch.LongTensor(y_val).to(DEVICE)

        # Build model
        self.model = LSTMRegimeModel(
            n_features=n_features,
            n_classes=self.n_regimes,
            hidden_size=self.hidden_size,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Model parameters: {n_params:,}")

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2
        )

        # Training loop — track balanced val accuracy for early stopping
        # Val set is recent data (~80% Expansion), so raw accuracy is misleading.
        # Balanced accuracy = mean per-class recall, rewards catching transitions.
        best_val_score = 0.0
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 30
        batch_size = 16
        best_state = None

        # Classes present in validation set
        val_classes = set(y_val.tolist())

        for epoch in range(200):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle training data (sequences, not individual time steps)
            perm = torch.randperm(len(X_train_t))
            X_train_shuffled = X_train_t[perm]
            y_train_shuffled = y_train_t[perm]

            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_shuffled[i:i + batch_size]
                batch_y = y_train_shuffled[i:i + batch_size]

                # Add Gaussian noise during training for regularization
                noise = torch.randn_like(batch_X) * 0.02
                batch_X_noisy = batch_X + noise

                logits, _ = self.model(batch_X_noisy)
                loss = criterion(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validation (no noise)
            self.model.eval()
            with torch.no_grad():
                val_logits, _ = self.model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_preds = val_logits.argmax(dim=-1).cpu().numpy()
                val_true = y_val_t.cpu().numpy()
                val_acc = (val_preds == val_true).mean()

                # Balanced accuracy: mean per-class recall
                per_class_acc = []
                for c in val_classes:
                    mask = val_true == c
                    if mask.sum() > 0:
                        per_class_acc.append((val_preds[mask] == c).mean())
                bal_acc = np.mean(per_class_acc) if per_class_acc else val_acc

            scheduler.step(epoch + val_loss * 0)  # CosineAnnealing uses epoch count

            # Early stopping on combined metric: 70% raw accuracy + 30% balanced
            # Raw accuracy matters because val set reflects real distribution
            val_score = 0.7 * val_acc + 0.3 * bal_acc
            if val_score > best_val_score or (val_score == best_val_score and val_loss < best_val_loss):
                best_val_score = val_score
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or patience_counter == 0:
                logger.info(
                    f"  Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, acc={val_acc:.3f}, "
                    f"bal_acc={bal_acc:.3f}, score={val_score:.3f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.1e}"
                )

            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        # Record training metadata
        self._training_metadata = {
            "epochs_trained": epoch + 1,
            "best_val_loss": best_val_loss,
            "best_bal_acc": best_val_score,
            "n_sequences": len(X_seq),
            "n_features": n_features,
            "n_params": n_params,
            "feature_cols": self.feature_cols,
        }

        self.is_fitted = True
        logger.info(
            f"  Neural net training complete. "
            f"Best bal_acc={best_val_score:.3f}, val_loss={best_val_loss:.4f}, epochs={epoch+1}"
        )
        return self

    def _create_labels(self, X_raw: pd.DataFrame, recession: pd.Series) -> pd.Series:
        """
        Create 5-class labels using the same logic as EnsembleRegimeClassifier.
        Imports and delegates to avoid code duplication.
        """
        from models.regime_classifier import EnsembleRegimeClassifier

        temp_classifier = EnsembleRegimeClassifier(n_regimes=self.n_regimes)
        temp_classifier._label_params = self._label_params
        X_unscaled = pd.DataFrame(
            X_raw.values, index=X_raw.index,
            columns=X_raw.columns,
        )
        # _create_5class_labels expects the base features only
        # but handles missing columns gracefully via 'if col in X_df.columns'
        labels = temp_classifier._create_5class_labels(X_unscaled, recession)
        return labels

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding-window sequences from the time series.
        Each sequence label is the label at the LAST time step in the window.
        """
        sequences, labels = [], []
        for i in range(self.seq_len, len(X)):
            sequences.append(X[i - self.seq_len:i])
            labels.append(y[i])
        return np.array(sequences), np.array(labels)

    def _oversample_minority(
        self, X: np.ndarray, y: np.ndarray, max_repeat: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Duplicate sequences of minority classes to reduce imbalance.
        Preserves temporal ordering within each sequence.
        Adds slight Gaussian noise to duplicates to prevent memorization.
        """
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        X_parts, y_parts = [X], [y]
        for cls, cnt in zip(classes, counts):
            if cnt < max_count // 3:  # oversample if < 1/3 of majority
                mask = y == cls
                repeat_times = min(max_count // max(cnt, 1), max_repeat) - 1
                if repeat_times > 0:
                    X_repeated = np.repeat(X[mask], repeat_times, axis=0)
                    # Add slight noise to duplicates
                    X_repeated += np.random.normal(0, 0.02, X_repeated.shape)
                    X_parts.append(X_repeated)
                    y_parts.append(np.repeat(y[mask], repeat_times))
                    logger.info(
                        f"  Oversampled class {cls} ({REGIME_LABELS.get(cls, '?')}): "
                        f"{cnt} -> {cnt + cnt * repeat_times}"
                    )

        return np.concatenate(X_parts), np.concatenate(y_parts)

    # -----------------------------------------------------------------
    #  Prediction
    # -----------------------------------------------------------------

    def predict_proba(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities.
        X_raw must have at least seq_len rows for the LSTM lookback.
        Returns DataFrame with columns = regime labels, index = dates.
        Only returns predictions for rows where a full sequence is available.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        # Select and align feature columns
        available_cols = [c for c in self.feature_cols if c in X_raw.columns]
        missing_cols = [c for c in self.feature_cols if c not in X_raw.columns]

        X_clean = X_raw[available_cols].copy()
        for col in missing_cols:
            X_clean[col] = 0.0
        X_clean = X_clean[self.feature_cols]

        X_clean = X_clean.ffill().bfill()
        if self._col_medians is not None:
            X_clean = X_clean.fillna(self._col_medians)
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X_scaled = self.scaler.transform(X_clean.values)
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequences
        X_seq, indices = [], []
        for i in range(self.seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i - self.seq_len:i])
            indices.append(X_clean.index[i])

        if not X_seq:
            # Not enough data for even one sequence — return empty
            return pd.DataFrame(
                columns=[REGIME_LABELS[i] for i in range(self.n_regimes)],
                dtype=float,
            )

        X_seq = np.array(X_seq)
        X_tensor = torch.FloatTensor(X_seq).to(DEVICE)

        # Store for feature importance computation
        self._last_input_tensor = X_tensor.detach().clone()

        self.model.eval()
        with torch.no_grad():
            logits, attn_weights = self.model(X_tensor)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            self._last_attn_weights = attn_weights.cpu().numpy()

        return pd.DataFrame(
            proba,
            index=indices,
            columns=[REGIME_LABELS[i] for i in range(self.n_regimes)],
        )

    def predict(self, X_raw: pd.DataFrame) -> pd.Series:
        """Return the most likely regime for each observation."""
        proba = self.predict_proba(X_raw)
        if proba.empty:
            return pd.Series(dtype=str, name="regime")
        regime_idx = proba.values.argmax(axis=1)
        return pd.Series(
            [REGIME_LABELS[i] for i in regime_idx],
            index=proba.index,
            name="regime",
        )

    def predict_single(self, feature_row: pd.DataFrame, history: pd.DataFrame) -> Dict:
        """
        Predict for a single current observation using trailing history.
        Used by the nowcast module.

        Args:
            feature_row: 1-row DataFrame with current feature values
            history: DataFrame with at least (seq_len - 1) rows of recent history

        Returns:
            Dict with regime, probabilities, confidence
        """
        if not self.is_fitted or self.model is None:
            return {"regime": "Unknown", "probabilities": {}, "confidence": 0.0}

        # Combine history + current row
        combined = pd.concat([history, feature_row])
        if len(combined) < self.seq_len:
            # Pad with repeated first row if not enough history
            pad_rows = self.seq_len - len(combined)
            padding = pd.concat([combined.iloc[:1]] * pad_rows)
            combined = pd.concat([padding, combined])

        # Use last seq_len rows
        combined = combined.iloc[-self.seq_len:]

        # Align columns
        available_cols = [c for c in self.feature_cols if c in combined.columns]
        missing_cols = [c for c in self.feature_cols if c not in combined.columns]
        X = combined[available_cols].copy()
        for col in missing_cols:
            X[col] = 0.0
        X = X[self.feature_cols].ffill().bfill()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X_scaled = self.scaler.transform(X.values)
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(DEVICE)

        # Store for feature importance
        self._last_input_tensor = X_tensor.detach().clone()

        self.model.eval()
        with torch.no_grad():
            logits, attn_weights = self.model(X_tensor)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            self._last_attn_weights = attn_weights.cpu().numpy()

        regime_idx = int(np.argmax(proba))
        probabilities = {}
        for i in range(self.n_regimes):
            probabilities[REGIME_LABELS[i]] = round(float(proba[i]), 4)

        return {
            "regime": REGIME_LABELS.get(regime_idx, "Unknown"),
            "probabilities": probabilities,
            "confidence": float(proba[regime_idx]),
        }

    # -----------------------------------------------------------------
    #  Feature Importance
    # -----------------------------------------------------------------

    def feature_importance(self) -> pd.Series:
        """
        Compute feature importance via gradient × attention weights.
        Uses the most recent prediction's input tensor.
        """
        if not self.is_fitted or self.model is None:
            return pd.Series(dtype=float)
        if self._last_input_tensor is None:
            return pd.Series(dtype=float)

        X = self._last_input_tensor.clone().requires_grad_(True).to(DEVICE)
        self.model.eval()

        logits, attn_weights = self.model(X)
        # Backprop through the predicted class logits
        pred_classes = logits.argmax(dim=-1)
        target = logits.gather(1, pred_classes.unsqueeze(1)).sum()
        target.backward()

        # Gradient w.r.t. input: (batch, seq_len, n_features)
        grad = X.grad.abs()

        # Weight by attention: (batch, seq_len, 1) * (batch, seq_len, n_features)
        attn = attn_weights.detach().unsqueeze(-1)  # (batch, seq_len, 1)
        weighted_grad = (grad * attn).mean(dim=(0, 1))  # (n_features,)

        importance = weighted_grad.detach().cpu().numpy()
        total = importance.sum()
        if total > 0:
            importance = importance / total

        return pd.Series(
            importance,
            index=self.feature_cols,
            name="importance",
        ).sort_values(ascending=False)

    # -----------------------------------------------------------------
    #  Persistence
    # -----------------------------------------------------------------

    def save(self, path: Optional[Path] = None):
        """Save the trained model to disk."""
        path = path or MODEL_DIR / "neural_regime_classifier.pt"
        if self.model is None:
            logger.warning("No model to save")
            return

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "n_features": len(self.feature_cols),
                "n_classes": self.n_regimes,
                "hidden_size": self.hidden_size,
                "n_layers": 1,
                "bidirectional": False,
            },
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "seq_len": self.seq_len,
            "n_regimes": self.n_regimes,
            "hidden_size": self.hidden_size,
            "col_medians": self._col_medians,
            "label_params": self._label_params,
            "training_metadata": self._training_metadata,
        }, path)
        logger.info(f"Neural regime classifier saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NeuralRegimeClassifier":
        """Load a trained model from disk."""
        path = path or MODEL_DIR / "neural_regime_classifier.pt"
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        instance = cls(
            n_regimes=checkpoint["n_regimes"],
            seq_len=checkpoint["seq_len"],
            hidden_size=checkpoint["hidden_size"],
        )

        config = checkpoint["model_config"]
        instance.model = LSTMRegimeModel(
            n_features=config["n_features"],
            n_classes=config["n_classes"],
            hidden_size=config["hidden_size"],
            n_layers=config.get("n_layers", 1),
        ).to(DEVICE)
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.model.eval()

        instance.scaler = checkpoint["scaler"]
        instance.feature_cols = checkpoint["feature_cols"]
        instance._col_medians = checkpoint["col_medians"]
        instance._label_params = checkpoint.get("label_params")
        instance._training_metadata = checkpoint.get("training_metadata", {})
        instance.is_fitted = True

        logger.info(f"Neural regime classifier loaded from {path}")
        return instance
