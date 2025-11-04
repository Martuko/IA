
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import shuffle as sk_shuffle

from joblib import Parallel, delayed

@dataclass
class ModelSpec:
    name: str
    algo: str           # 'logistic' or 'svm'
    params: Dict[str, Any]
    model: SGDClassifier = field(default=None, repr=False)

def _make_sgd_model(spec: ModelSpec, random_state: int, n_jobs: int = -1) -> SGDClassifier:
    if spec.algo == "logistic":
        # Softmax via 'log_loss' (multiclass)
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=spec.params.get("alpha", 1e-4),
            learning_rate=spec.params.get("learning_rate", "optimal"),
            eta0=spec.params.get("eta0", 0.0),
            power_t=spec.params.get("power_t", 0.5),
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif spec.algo == "svm":
        return SGDClassifier(
            loss=spec.params.get("loss", "hinge"),  # 'hinge' or 'modified_huber'
            penalty="l2",
            alpha=spec.params.get("alpha", 1e-4),
            learning_rate=spec.params.get("learning_rate", "optimal"),
            eta0=spec.params.get("eta0", 0.0),
            power_t=spec.params.get("power_t", 0.5),
            random_state=random_state,
            n_jobs=n_jobs
        )
    else:
        raise ValueError(f"Algo desconocido: {spec.algo}")

def _train_one_epoch(spec: ModelSpec, Xtr, ytr, classes, batch_size: int, epoch: int) -> ModelSpec:
    # partial_fit over mini-batches
    n = Xtr.shape[0]
    # Shuffle indices each epoch
    idx = np.random.permutation(n)
    first_call = (epoch == 0)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = Xtr[idx[start:end]]
        yb = ytr[idx[start:end]]
        if first_call:
            spec.model.partial_fit(Xb, yb, classes=classes)
            first_call = False
        else:
            spec.model.partial_fit(Xb, yb)
    return spec

def _metric(y_true, y_pred, name: str) -> float:
    if name == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif name == "f1_macro":
        return f1_score(y_true, y_pred, average="macro")
    else:
        raise ValueError(f"Métrica '{name}' no soportada")

def train_with_elimination(cfg: Dict[str, Any], Xtr, ytr, Xte, yte, class_names: List[str], n_jobs: int = -1):
    random_state = cfg.get("random_state", 42)
    sup = cfg["supervised"]
    n_epochs = sup["n_epochs"]
    eval_every = sup["eval_every_epochs"]
    batch_size = sup["batch_size"]
    metric_name = sup.get("metric", "f1_macro")

    # Label encoding for y
    le = LabelEncoder()
    ytr_enc = le.fit_transform(ytr)
    yte_enc = le.transform(yte)
    classes = np.unique(ytr_enc)

    # Build specs
    specs: List[ModelSpec] = []
    for p in sup["logistic"]["configs"]:
        specs.append(ModelSpec(name=p["name"], algo="logistic", params=p))
    for p in sup["svm"]["configs"]:
        specs.append(ModelSpec(name=p["name"], algo="svm", params=p))

    # Instantiate models
    for s in specs:
        s.model = _make_sgd_model(s, random_state=random_state, n_jobs=n_jobs)

    active: List[ModelSpec] = specs.copy()
    history = []  # to record metrics over time

    for epoch in range(n_epochs):
        # Train all active models in parallel (each over all mini-batches)
        active = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_train_one_epoch)(s, Xtr, ytr_enc, classes, batch_size, epoch) for s in active
        )

        # Evaluate only on training set at intervals
        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            eval_rows = []
            preds = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(lambda m: m.model.predict(Xtr))(s) for s in active
            )
            for s, y_pred in zip(active, preds):
                score = _metric(ytr_enc, y_pred, metric_name)
                eval_rows.append({"epoch": epoch + 1, "name": s.name, "algo": s.algo, "metric": metric_name, "score": float(score)})
            df_eval = pd.DataFrame(eval_rows).sort_values(by="score", ascending=False).reset_index(drop=True)
            history.append(df_eval)

            # Eliminate worst-performing config across all remaining (ensure at least 2 remain)
            if len(active) > 2 and (epoch + 1) % eval_every == 0:
                worst_name = df_eval.iloc[-1]["name"]
                active = [s for s in active if s.name != worst_name]

    # Final train-set leaderboard
    final_eval = history[-1].copy() if history else pd.DataFrame(columns=["epoch","name","algo","metric","score"])

    # Select top-2 on final train metric
    top2_names = final_eval["name"].head(2).tolist()
    top2 = [s for s in specs if s.name in top2_names]

    # Evaluate top-2 on test set
    test_rows = []
    for s in top2:
        y_pred_test = s.model.predict(Xte)
        acc = accuracy_score(yte_enc, y_pred_test)
        f1m = f1_score(yte_enc, y_pred_test, average="macro")
        report = classification_report(yte_enc, y_pred_test, target_names=[str(c) for c in le.classes_], output_dict=True, zero_division=0)
        test_rows.append({
            "name": s.name,
            "algo": s.algo,
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1m),
            "classification_report": report
        })

    return pd.concat(history, ignore_index=True) if history else final_eval, pd.DataFrame(test_rows), active
