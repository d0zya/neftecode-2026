from __future__ import annotations

import argparse
from pathlib import Path
import copy

import pandas as pd
import torch
from torch.utils.data import DataLoader

from training_pipeline import (
    ARTIFACTS_DIR,
    DeepSetSumRegressor,
    ScenarioSetDataset,
    TRAIN_TARGETS,
    build_categorical_encoders,
    collate_set_batch,
    filter_categorical_features_by_train_coverage,
    filter_component_features_by_train_coverage,
    fit_scalers,
    load_data,
    predict_on_test,
    prepare_component_features,
    set_global_seed,
)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def prepare_inference_context() -> dict[str, object]:
    mixtures_train, mixtures_test, properties = load_data()
    (
        _,
        X_feature_df,
        X_mask_df,
        typical_features_map,
        typical_masks_map,
        component_feature_names,
        categorical_feature_df,
        typical_categorical_map,
        categorical_feature_names,
    ) = prepare_component_features(properties)
    (
        X_feature_df,
        X_mask_df,
        typical_features_map,
        typical_masks_map,
        component_feature_names,
        _,
    ) = filter_component_features_by_train_coverage(
        mixtures_train=mixtures_train,
        X_feature_df=X_feature_df,
        X_mask_df=X_mask_df,
        typical_features_map=typical_features_map,
        typical_masks_map=typical_masks_map,
        min_coverage=0.12,
    )
    (
        categorical_feature_df,
        typical_categorical_map,
        categorical_feature_names,
        _,
        _,
    ) = filter_categorical_features_by_train_coverage(
        mixtures_train=mixtures_train,
        categorical_feature_df=categorical_feature_df,
        typical_categorical_map=typical_categorical_map,
        min_coverage=0.08,
        max_cardinality=20,
    )
    categorical_encoders = build_categorical_encoders(categorical_feature_df)
    scaler_temp, scaler_time, scaler_biofuel, le_catalyst, scaler_catalyst, scaler_y = fit_scalers(mixtures_train)

    return {
        "mixtures_train": mixtures_train,
        "mixtures_test": mixtures_test,
        "X_feature_df": X_feature_df,
        "X_mask_df": X_mask_df,
        "typical_features_map": typical_features_map,
        "typical_masks_map": typical_masks_map,
        "component_feature_names": component_feature_names,
        "categorical_feature_df": categorical_feature_df,
        "typical_categorical_map": typical_categorical_map,
        "categorical_feature_names": categorical_feature_names,
        "categorical_encoders": categorical_encoders,
        "scaler_temp": scaler_temp,
        "scaler_time": scaler_time,
        "scaler_biofuel": scaler_biofuel,
        "le_catalyst": le_catalyst,
        "scaler_catalyst": scaler_catalyst,
        "scaler_y": scaler_y,
    }


def validate_checkpoint_compatibility(checkpoint: dict[str, object], context: dict[str, object]) -> None:
    feature_state = checkpoint.get("feature_state")
    if not isinstance(feature_state, dict):
        return {}
    expected_component = feature_state.get("component_feature_names")
    current_component = context.get("component_feature_names")
    if expected_component is not None and current_component is not None:
        if set(expected_component) != set(current_component):
            raise ValueError("Checkpoint numeric feature set does not match current preprocessing.")

    expected_categorical = feature_state.get("categorical_feature_names")
    current_categorical = context.get("categorical_feature_names")
    if expected_categorical is not None and current_categorical is not None:
        if set(expected_categorical) != set(current_categorical):
            raise ValueError("Checkpoint categorical feature set does not match current preprocessing.")

    return feature_state


def build_test_loader(context: dict[str, object], batch_size: int) -> tuple[list[str], DataLoader]:
    mixtures_test = context["mixtures_test"]
    test_scenario_ids = mixtures_test["scenario_id"].unique().tolist()
    test_dataset = ScenarioSetDataset(
        mixtures=mixtures_test,
        scenario_ids=test_scenario_ids,
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_set_batch)
    return test_scenario_ids, test_loader


def reorder_context_for_checkpoint(base_context: dict[str, object],checkpoint_component_order: list[str] | None,
                                   checkpoint_categorical_order: list[str] | None,) -> dict[str, object]:
    ctx = copy.deepcopy(base_context)
    if checkpoint_component_order is not None:
        current_order = base_context["component_feature_names"]
        if current_order != checkpoint_component_order:
            ctx["X_feature_df"] = ctx["X_feature_df"][checkpoint_component_order]
            ctx["X_mask_df"] = ctx["X_mask_df"][checkpoint_component_order]
            ctx["component_feature_names"] = checkpoint_component_order
            indices = [current_order.index(col) for col in checkpoint_component_order]
            for comp in list(ctx["typical_features_map"].keys()):
                old_arr = ctx["typical_features_map"][comp]
                ctx["typical_features_map"][comp] = old_arr[indices]
            for comp in list(ctx["typical_masks_map"].keys()):
                old_arr = ctx["typical_masks_map"][comp]
                ctx["typical_masks_map"][comp] = old_arr[indices]

    if checkpoint_categorical_order is not None:
        current_cat = base_context.get("categorical_feature_names", [])
        if current_cat != checkpoint_categorical_order:
            ctx["categorical_feature_df"] = ctx["categorical_feature_df"][checkpoint_categorical_order]
            ctx["categorical_feature_names"] = checkpoint_categorical_order
            indices_cat = [current_cat.index(col) for col in checkpoint_categorical_order]
            for comp in list(ctx["typical_categorical_map"].keys()):
                old_arr = ctx["typical_categorical_map"][comp]
                ctx["typical_categorical_map"][comp] = old_arr[indices_cat]
    return ctx


def run_inference(
    artifacts_dir: Path,
    output_path: Path,
    batch_size: int = 16,
    device_name: str = "auto",
    seed: int = 42,
) -> pd.DataFrame:
    set_global_seed(seed)
    device = resolve_device(device_name)
    base_context = prepare_inference_context() 

    checkpoint_paths = sorted(artifacts_dir.glob("deepset_sum_*.pt"))
    if len(checkpoint_paths) != len(TRAIN_TARGETS):
        raise FileNotFoundError(
            f"Expected {len(TRAIN_TARGETS)} checkpoints in {artifacts_dir}, found {len(checkpoint_paths)}."
        )

    predictions_by_target = {}
    test_scenario_ids, _ = build_test_loader(base_context, batch_size=batch_size)

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        feature_state  = validate_checkpoint_compatibility(checkpoint, base_context)
        
        ctx = reorder_context_for_checkpoint(
            base_context,
            checkpoint_component_order=feature_state.get("component_feature_names") if feature_state else None,
            checkpoint_categorical_order=feature_state.get("categorical_feature_names") if feature_state else None,
        )
        _, test_loader = build_test_loader(ctx, batch_size=batch_size)
        
        hyperparameters = checkpoint.get("hyperparameters", checkpoint["model_kwargs"])
        model = DeepSetSumRegressor(**hyperparameters).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        target_index = int(checkpoint["target_index"])
        target_name = str(checkpoint["target_name"])
        predictions = predict_on_test(
            model=model,
            loader=test_loader,
            device=device,
            scaler_y=ctx["scaler_y"],
            target_index=target_index,
        )
        predictions_by_target[target_name] = predictions

    results_df = pd.DataFrame(
        {
            "scenario_id": test_scenario_ids,
            TRAIN_TARGETS[0]: predictions_by_target[TRAIN_TARGETS[0]],
            TRAIN_TARGETS[1]: predictions_by_target[TRAIN_TARGETS[1]],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic inference from saved Deep Sets checkpoints.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictions_df = run_inference(
        artifacts_dir=args.artifacts_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        device_name=args.device,
        seed=args.seed,
    )
    print(f"Saved predictions to: {args.output}")
    print(predictions_df.head().to_string(index=False))
