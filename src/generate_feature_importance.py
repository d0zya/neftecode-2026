from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from training_pipeline import (
    ARTIFACTS_DIR,
    PROJECT_DIR,
    DeepSetSumRegressor,
    ScenarioSetDataset,
    TRAIN_TARGETS,
    collate_set_batch,
    make_target_slug,
    move_batch_to_device,
    set_global_seed,
)
from inference_from_artifacts import prepare_inference_context, resolve_device


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "feature_importance"


def make_file_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Zа-яА-Я0-9]+", "_", value).strip("_").lower()
    return slug or "group"


def component_group(component_name: str) -> str:
    return re.sub(r"_\d+$", "", component_name)


def build_train_loader(context: dict[str, object], batch_size: int) -> DataLoader:
    mixtures_train = context["mixtures_train"]
    train_scenario_ids = mixtures_train["scenario_id"].unique().tolist()
    dataset = ScenarioSetDataset(
        mixtures=mixtures_train,
        scenario_ids=train_scenario_ids,
        X_feature_df=context["X_feature_df"],
        X_mask_df=context["X_mask_df"],
        typical_features_map=context["typical_features_map"],
        typical_masks_map=context["typical_masks_map"],
        categorical_feature_df=context["categorical_feature_df"],
        typical_categorical_map=context["typical_categorical_map"],
        categorical_encoders=context["categorical_encoders"],
        scaler_temp=context["scaler_temp"],
        scaler_time=context["scaler_time"],
        scaler_biofuel=context["scaler_biofuel"],
        le_catalyst=context["le_catalyst"],
        scaler_catalyst=context["scaler_catalyst"],
        scaler_y=None,
        target_index=None,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_set_batch)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[DeepSetSumRegressor, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_kwargs = checkpoint.get("hyperparameters", checkpoint["model_kwargs"])
    model = DeepSetSumRegressor(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def collect_gradient_attributions(
    model: torch.nn.Module,
    loader: DataLoader,
    mixtures_train: pd.DataFrame,
    feature_names: list[str],
    target_scale: float,
    device: torch.device,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        component_values = batch["component_values"].detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        outputs = model(
            component_values,
            batch["feature_masks"],
            batch["component_cat_ids"],
            batch["mass_fractions"],
            batch["component_mask"],
            batch["conditions"],
        )
        outputs.sum().backward()

        gradients = component_values.grad.detach().cpu().numpy()
        values = component_values.detach().cpu().numpy()
        masks = batch["feature_masks"].detach().cpu().numpy()
        component_mask = batch["component_mask"].detach().cpu().numpy()
        scenario_ids = list(batch["scenario_ids"])

        for batch_idx, scenario_id in enumerate(scenario_ids):
            scenario_components = mixtures_train[mixtures_train["scenario_id"] == scenario_id]["Компонент"].tolist()
            component_count = min(len(scenario_components), values.shape[1])
            for component_idx in range(component_count):
                if component_mask[batch_idx, component_idx] == 0:
                    continue
                comp_name = scenario_components[component_idx]
                group_name = component_group(comp_name)
                for feature_idx, feature_name in enumerate(feature_names):
                    if masks[batch_idx, component_idx, feature_idx] == 0:
                        continue
                    feature_value = float(values[batch_idx, component_idx, feature_idx])
                    attribution = float(gradients[batch_idx, component_idx, feature_idx] * feature_value * target_scale)
                    records.append(
                        {
                            "scenario_id": scenario_id,
                            "component": comp_name,
                            "component_group": group_name,
                            "feature": feature_name,
                            "feature_value": feature_value,
                            "attribution": attribution,
                            "abs_attribution": abs(attribution),
                        }
                    )

    return pd.DataFrame.from_records(records)


def select_top_features(group_df: pd.DataFrame, max_features: int) -> list[str]:
    importance = (
        group_df.groupby("feature", as_index=False)["abs_attribution"]
        .mean()
        .sort_values("abs_attribution", ascending=False)
    )
    return importance["feature"].head(max_features).tolist()


def plot_beeswarm_summary(
    group_df: pd.DataFrame,
    group_name: str,
    target_label: str,
    output_path: Path,
    max_features: int,
    random_seed: int,
) -> None:
    top_features = select_top_features(group_df, max_features=max_features)
    if not top_features:
        return

    plot_df = group_df[group_df["feature"].isin(top_features)].copy()
    rng = np.random.default_rng(random_seed)

    fig_height = max(7.0, 0.48 * len(top_features) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    cmap = plt.get_cmap("cool")
    y_positions = np.arange(len(top_features))

    for y_idx, feature_name in enumerate(top_features):
        feature_df = plot_df[plot_df["feature"] == feature_name]
        x_values = feature_df["attribution"].to_numpy(dtype=float)
        color_values = feature_df["feature_value"].to_numpy(dtype=float)

        if len(x_values) == 0:
            continue

        if np.nanmax(color_values) > np.nanmin(color_values):
            colors = (color_values - np.nanmin(color_values)) / (np.nanmax(color_values) - np.nanmin(color_values))
        else:
            colors = np.full_like(color_values, 0.5, dtype=float)

        jitter = rng.normal(loc=0.0, scale=0.07, size=len(x_values))
        jitter = np.clip(jitter, -0.24, 0.24)
        ax.scatter(
            x_values,
            np.full(len(x_values), y_idx) + jitter,
            c=colors,
            cmap=cmap,
            s=26,
            alpha=0.9,
            linewidths=0,
        )

    ax.axvline(0.0, color="#7a7a7a", linewidth=1.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_features, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP-like value (impact on model output)", fontsize=14)
    ax.set_title(f"SHAP-like summary for group: {group_name} ({target_label})", fontsize=16, pad=14)
    ax.grid(axis="y", color="#eeeeee", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", color="#eeeeee", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Feature value", rotation=90, labelpad=16, fontsize=13)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_importance_outputs(
    attributions_df: pd.DataFrame,
    output_dir: Path,
    target_name: str,
    target_index: int,
    max_features: int,
    random_seed: int,
) -> None:
    target_slug = make_target_slug(target_name)
    target_label = f"Target{target_index + 1}"
    target_dir = output_dir / target_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    attributions_df.to_csv(target_dir / "attributions.csv", index=False)
    summary_df = (
        attributions_df.groupby(["component_group", "feature"], as_index=False)["abs_attribution"]
        .mean()
        .sort_values(["component_group", "abs_attribution"], ascending=[True, False])
    )
    summary_df.to_csv(target_dir / "summary.csv", index=False)

    for group_name, group_df in attributions_df.groupby("component_group"):
        if group_df.empty:
            continue
        output_path = target_dir / f"{make_file_slug(group_name)}.png"
        plot_beeswarm_summary(
            group_df=group_df,
            group_name=group_name,
            target_label=target_label,
            output_path=output_path,
            max_features=max_features,
            random_seed=random_seed + target_index,
        )


def generate_feature_importance(
    artifacts_dir: Path = ARTIFACTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 8,
    device_name: str = "auto",
    max_features: int = 20,
    seed: int = 42,
) -> None:
    set_global_seed(seed)
    device = resolve_device(device_name)
    context = prepare_inference_context()

    feature_names = list(context["component_feature_names"])
    loader = build_train_loader(context, batch_size=batch_size)
    checkpoint_paths = sorted(artifacts_dir.glob("deepset_sum_*.pt"))
    if len(checkpoint_paths) != len(TRAIN_TARGETS):
        raise FileNotFoundError(
            f"Expected {len(TRAIN_TARGETS)} checkpoints in {artifacts_dir}, found {len(checkpoint_paths)}."
        )

    for checkpoint_path in checkpoint_paths:
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
        target_name = str(checkpoint["target_name"])
        target_index = int(checkpoint["target_index"])
        target_scale = float(context["scaler_y"].scale_[target_index])

        attributions_df = collect_gradient_attributions(
            model=model,
            loader=loader,
            mixtures_train=context["mixtures_train"],
            feature_names=feature_names,
            target_scale=target_scale,
            device=device,
        )
        save_importance_outputs(
            attributions_df=attributions_df,
            output_dir=output_dir,
            target_name=target_name,
            target_index=target_index,
            max_features=max_features,
            random_seed=seed,
        )
        print(f"Saved feature importance for {target_name} to: {output_dir / make_target_slug(target_name)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP-like feature attribution plots from saved checkpoints.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_feature_importance(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        max_features=args.max_features,
        seed=args.seed,
    )
