from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from joblib import Parallel, delayed


# =========================
#   ESTRUCTURAS / MODELOS
# =========================

@dataclass
class ModelSpec:
    name: str
    algo: str           # 'logistic' o 'svm'
    params: Dict[str, Any]
    model: SGDClassifier = field(default=None, repr=False)


def _make_sgd_model(spec: ModelSpec, random_state: int) -> SGDClassifier:
    """Crea un SGDClassifier para logística (softmax) o SVM lineal."""
    if spec.algo == "logistic":
        return SGDClassifier(
            loss="log_loss",       # regresión logística multinomial
            penalty="l2",
            alpha=spec.params.get("alpha", 1e-4),
            learning_rate=spec.params.get("learning_rate", "optimal"),
            eta0=spec.params.get("eta0", 0.0),
            power_t=spec.params.get("power_t", 0.5),
            random_state=random_state,
        )
    elif spec.algo == "svm":
        return SGDClassifier(
            loss=spec.params.get("loss", "hinge"),   # 'hinge' o 'modified_huber'
            penalty="l2",
            alpha=spec.params.get("alpha", 1e-4),
            learning_rate=spec.params.get("learning_rate", "optimal"),
            eta0=spec.params.get("eta0", 0.0),
            power_t=spec.params.get("power_t", 0.5),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Algo desconocido: {spec.algo}")


# =========================
#   MINI-BATCHES BALANCEADOS
# =========================

def _balanced_minibatches_idx(
    y: np.ndarray,
    batch_size: int,
    classes: np.ndarray,
    rng: np.random.Generator
) -> List[np.ndarray]:
    """
    Genera índices de mini-batches con balance por clase (con reposición para las minorías).
    Útil cuando hay clases MUY desbalanceadas.
    """
    n_classes = len(classes)
    per_class = max(1, batch_size // n_classes)
    cls_to_idx = {c: np.where(y == c)[0] for c in classes}

    max_len = max(len(v) for v in cls_to_idx.values())
    n_batches = int(np.ceil(max_len / per_class))

    batches = []
    for _ in range(n_batches):
        parts = []
        for c in classes:
            pool = cls_to_idx[c]
            if len(pool) == 0:
                continue
            chosen = rng.choice(pool, size=per_class, replace=True)
            parts.append(chosen)
        if not parts:
            continue
        batch = np.concatenate(parts)
        rng.shuffle(batch)
        # ajustar tamaño
        if batch.size > batch_size:
            batch = batch[:batch_size]
        elif batch.size < batch_size:
            # completa con la mayoritaria (1ª clase)
            add = rng.choice(cls_to_idx[classes[0]], size=(batch_size - batch.size), replace=True)
            batch = np.concatenate([batch, add])
            rng.shuffle(batch)
        batches.append(batch)
    return batches


# =========================
#   ENTRENAMIENTO 1 ÉPOCA
# =========================

def _train_one_epoch(
    spec: ModelSpec,
    Xtr,
    ytr: np.ndarray,
    classes: np.ndarray,
    batch_size: int,
    epoch: int,
    class_weight_map: Optional[Dict[int, float]],
    balanced_minibatches: bool,
    gamma: float,
    rng: np.random.Generator,
) -> ModelSpec:
    """
    Entrena UNA época con partial_fit.
    - si balanced_minibatches=True: lotes más equilibrados
    - si class_weight_map != None: aplica sample_weight templado con gamma
    """
    if balanced_minibatches:
        batches = _balanced_minibatches_idx(ytr, batch_size, classes, rng)
    else:
        n = Xtr.shape[0]
        idx = rng.permutation(n)
        batches = [idx[i:i + batch_size] for i in range(0, n, batch_size)]

    first_call = (epoch == 0)

    for idx in batches:
        Xb = Xtr[idx]
        yb = ytr[idx]

        sw = None
        if class_weight_map is not None:
            # pesos balanceados puros
            w_bal = np.vectorize(class_weight_map.get)(yb).astype(float)
            # mezcla templada: 1 * (1 - gamma) + w_bal * gamma
            sw = (1.0 - gamma) * 1.0 + gamma * w_bal
            # normalizar al tamaño del batch
            sw *= (len(yb) / np.sum(sw))
            # evitar pesos absurdos
            np.clip(sw, 0.5, 5.0, out=sw)

        if first_call:
            spec.model.partial_fit(Xb, yb, classes=classes, sample_weight=sw)
            first_call = False
        else:
            spec.model.partial_fit(Xb, yb, sample_weight=sw)

    return spec


# =========================
#   MÉTRICA
# =========================

def _metric(y_true, y_pred, name: str) -> float:
    if name == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif name == "f1_macro":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    else:
        raise ValueError(f"Métrica '{name}' no soportada")


# =========================
#   PIPELINE PRINCIPAL
# =========================

def train_with_elimination(
    cfg: Dict[str, Any],
    Xtr,
    ytr,
    Xte,
    yte,
    class_names: List[str] | None,
    n_jobs: int = -1
):
    """
    - crea todas las configs (logistic + svm)
    - entrena por épocas
    - cada eval_every_epochs elimina la peor
    - al final evalúa en test las 2 mejores
    """
    random_state = cfg.get("random_state", 42)
    sup = cfg["supervised"]
    n_epochs = sup["n_epochs"]
    eval_every = sup["eval_every_epochs"]
    batch_size = sup["batch_size"]
    metric_name = sup.get("metric", "f1_macro")

    # nuevos flags
    use_sw = sup.get("use_sample_weights", True)
    balanced_minibatches = sup.get("balanced_minibatches", False)  # mejor False en tu caso
    gamma = float(sup.get("class_weight_gamma", 0.6))  # 0=no pesar, 1=peso full

    rng = np.random.default_rng(random_state)

    # --- encoding de etiquetas ---
    le = LabelEncoder()
    ytr_enc = le.fit_transform(ytr)
    yte_enc = le.transform(yte)
    classes = np.unique(ytr_enc)

    # --- construir specs ---
    specs: List[ModelSpec] = []
    for p in sup["logistic"]["configs"]:
        specs.append(ModelSpec(name=p["name"], algo="logistic", params=p))
    for p in sup["svm"]["configs"]:
        specs.append(ModelSpec(name=p["name"], algo="svm", params=p))

    # --- instanciar modelos ---
    for s in specs:
        s.model = _make_sgd_model(s, random_state=random_state)

    # --- pesos de clase (templados luego con gamma) ---
    class_weight_map = None
    if use_sw:
        cw = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=ytr_enc
        )
        class_weight_map = {cls: w for cls, w in zip(classes, cw)}

    active: List[ModelSpec] = specs.copy()
    history: List[pd.DataFrame] = []

    # =========================
    #   LOOP DE ENTRENAMIENTO
    # =========================
    for epoch in range(n_epochs):
        # entrenar todas las activas en paralelo
        active = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_train_one_epoch)(
                s, Xtr, ytr_enc, classes,
                batch_size,
                epoch,
                class_weight_map,
                balanced_minibatches,
                gamma,
                rng
            ) for s in active
        )

        # evaluación cada cierto número de épocas
        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            eval_rows = []

            preds = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(lambda m: m.model.predict(Xtr))(s) for s in active
            )

            for s, y_pred in zip(active, preds):
                score = _metric(ytr_enc, y_pred, metric_name)
                eval_rows.append({
                    "epoch": epoch + 1,
                    "name": s.name,
                    "algo": s.algo,
                    "metric": metric_name,
                    "score": float(score)
                })

            df_eval = pd.DataFrame(eval_rows).sort_values(by="score", ascending=False).reset_index(drop=True)
            history.append(df_eval)

            # eliminar la peor si hay más de 2
            if len(active) > 2 and (epoch + 1) % eval_every == 0:
                worst_name = df_eval.iloc[-1]["name"]
                active = [s for s in active if s.name != worst_name]

    # =========================
    #   EVAL EN TEST (TOP-2)
    # =========================
    final_eval = history[-1].copy() if history else pd.DataFrame(columns=["epoch","name","algo","metric","score"])
    top2_names = final_eval["name"].head(2).tolist()
    top2 = [s for s in specs if s.name in top2_names]

    test_rows = []
    for s in top2:
        y_pred_test = s.model.predict(Xte)
        acc = accuracy_score(yte_enc, y_pred_test)
        f1m = f1_score(yte_enc, y_pred_test, average="macro", zero_division=0)
        report = classification_report(
            yte_enc,
            y_pred_test,
            target_names=[str(c) for c in le.classes_],
            output_dict=True,
            zero_division=0
        )
        test_rows.append({
            "name": s.name,
            "algo": s.algo,
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1m),
            "classification_report": report,
        })

    history_df = pd.concat(history, ignore_index=True) if history else final_eval
    return history_df, pd.DataFrame(test_rows), active
