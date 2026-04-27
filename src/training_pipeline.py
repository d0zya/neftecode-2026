from __future__ import annotations

from copy import deepcopy
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
RANDOM_SEED = 42

TRAIN_TARGETS = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]

CATEGORICAL_CANDIDATE_COLUMNS = [
    "Класс субстрата",
    "Структура УВ-радикала",
    "Тип спиртового радикала",
    "Разветвленность радикала / радикалов",
    "Происхождение",
    "Тип АО",
    "Класс полиамина",
    "Модификация",
    "Тип сукцинимида",
    "Категория",
    "Тип лиганда",
    "Тип полимера",
]


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_target_slug(target_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", target_name).strip("_").lower()
    return slug or "target"


def save_model_artifact(
    save_path: Path,
    model: nn.Module,
    model_kwargs: dict[str, object],
    target_name: str,
    target_index: int,
    feature_state: dict[str, object],
    training_state: dict[str, object],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_name": target_name,
        "target_index": target_index,
        "model_class": model.__class__.__name__,
        "model_kwargs": model_kwargs,
        "model_state_dict": model.state_dict(),
        "feature_state": feature_state,
        "training_state": training_state,
    }
    torch.save(payload, save_path)

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mixtures_train = pd.read_csv(DATA_DIR / "daimler_mixtures_train.csv", encoding="utf-8-sig")
    mixtures_test = pd.read_csv(DATA_DIR / "daimler_mixtures_test.csv", encoding="utf-8-sig")
    properties = pd.read_csv(DATA_DIR / "daimler_component_properties.csv", encoding="utf-8-sig")
    return mixtures_train, mixtures_test, properties


def prepare_component_features(
    properties: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[str],
    pd.DataFrame,
    dict[str, np.ndarray],
    list[str],
]:
    properties = properties.copy()
    properties["component_group"] = properties["Компонент"].apply(lambda x: re.sub(r"_\d+$", "", x))
    properties["Значение показателя"] = (
        properties["Значение показателя"]
        .astype(str)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
        .str.replace("≤", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("°C", "", regex=False)
    )

    component_group_properties: dict[str, set[str]] = {}
    numeric_indicators: list[str] = []

    properties.loc[
        (properties["Наименование показателя"] == "Содержание других элементов (например, бора)")
        & (properties["Значение показателя"] == "0.05 (B)"),
        "Значение показателя",
    ] = 0.05
    properties.loc[
        (properties["Наименование показателя"] == "Щелочное число, ASTM D2896")
        & (properties["Значение показателя"] == "300-340"),
        "Значение показателя",
    ] = 320
    properties.loc[
        (properties["Наименование показателя"] == "Содержание металла (Ca/Mg), % масс.")
        & (properties["Значение показателя"] == "9.0 (Mg)"),
        "Значение показателя",
    ] = 9.0
    properties.loc[
        (properties["Наименование показателя"] == "Плотность при 15°С, ASTM D4052")
        & (properties["Значение показателя"] == "0.98-1.05"),
        "Значение показателя",
    ] = 1.015

    for group_name in properties["component_group"].unique():
        group_df = properties[properties["component_group"] == group_name]
        all_indicators = group_df["Наименование показателя"].dropna().unique()
        valid: list[str] = []

        for indicator in all_indicators:
            values = group_df[group_df["Наименование показателя"] == indicator]["Значение показателя"]
            if values.isna().all():
                continue

            numeric_vals = pd.to_numeric(values, errors="coerce")
            if not numeric_vals.isna().any() and (numeric_vals == 0).all():
                continue

            valid.append(indicator)
            if not numeric_vals.isna().any():
                numeric_indicators.append(indicator)

        component_group_properties[group_name] = set(valid)

    numerical_feature_columns = list(set(numeric_indicators))

    pivot_df = properties.pivot_table(
        index=["Компонент", "Наименование партии"],
        columns="Наименование показателя",
        values="Значение показателя",
        aggfunc="first",
    ).reset_index()

    pivot_df["component_group"] = pivot_df["Компонент"].apply(lambda x: re.sub(r"[_\d]+$", "", x))
    prop_cols = [c for c in pivot_df.columns if c not in ["Компонент", "Наименование партии", "component_group"]]

    for comp in pivot_df["Компонент"].unique():
        comp_mask = pivot_df["Компонент"] == comp
        typical_rows = pivot_df[comp_mask & (pivot_df["Наименование партии"] == "typical")]
        if typical_rows.empty:
            continue

        typical_row = typical_rows.iloc[0]
        group_name = pivot_df.loc[comp_mask, "component_group"].iloc[0]
        allowed_props = component_group_properties.get(group_name, set())

        for prop in prop_cols:
            if prop not in allowed_props:
                continue
            typical_val = typical_row.get(prop)
            if pd.isna(typical_val):
                continue
            fill_mask = comp_mask & (pivot_df["Наименование партии"] != "typical") & pivot_df[prop].isna()
            pivot_df.loc[fill_mask, prop] = typical_val

    pivot_df.set_index(["Компонент", "Наименование партии"], inplace=True)
    pivot_df = pivot_df.dropna(axis=1, how="all")

    component_feature_names = sorted(
        [
            column
            for column in pivot_df.columns
            if column in numerical_feature_columns
        ]
    )
    raw_feature_df = pivot_df[component_feature_names].apply(pd.to_numeric, errors="coerce")
    filled_feature_df = raw_feature_df.fillna(0.0).astype(np.float32)
    feature_mask_df = raw_feature_df.notna().astype(np.float32)

    typical_features_map: dict[str, np.ndarray] = {}
    typical_masks_map: dict[str, np.ndarray] = {}

    typical_mask = pivot_df.index.get_level_values("Наименование партии") == "typical"
    typical_features_df = filled_feature_df.loc[typical_mask]
    typical_mask_df = feature_mask_df.loc[typical_mask]

    for comp_name in typical_features_df.index.get_level_values("Компонент").unique():
        typical_features_map[comp_name] = typical_features_df.loc[(comp_name, "typical")].values.astype(np.float32)
        typical_masks_map[comp_name] = typical_mask_df.loc[(comp_name, "typical")].values.astype(np.float32)

    categorical_feature_names = sorted(
        [
            column
            for column in pivot_df.columns
            if column in CATEGORICAL_CANDIDATE_COLUMNS
        ]
    )
    if categorical_feature_names:
        categorical_feature_df = (
            pivot_df[categorical_feature_names]
            .fillna("__MISSING__")
            .astype(str)
            .replace({"nan": "__MISSING__", "None": "__MISSING__"})
        )
    else:
        categorical_feature_df = pd.DataFrame(index=pivot_df.index)

    typical_categorical_map: dict[str, np.ndarray] = {}
    if categorical_feature_names:
        typical_categorical_df = categorical_feature_df.loc[typical_mask]
        for comp_name in typical_categorical_df.index.get_level_values("Компонент").unique():
            typical_categorical_map[comp_name] = typical_categorical_df.loc[(comp_name, "typical")].values.astype(object)

    return (
        pivot_df,
        filled_feature_df,
        feature_mask_df,
        typical_features_map,
        typical_masks_map,
        component_feature_names,
        categorical_feature_df,
        typical_categorical_map,
        categorical_feature_names,
    )


def filter_component_features_by_train_coverage(
    mixtures_train: pd.DataFrame,
    X_feature_df: pd.DataFrame,
    X_mask_df: pd.DataFrame,
    typical_features_map: dict[str, np.ndarray],
    typical_masks_map: dict[str, np.ndarray],
    min_coverage: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], list[str], pd.Series]:
    feature_names = list(X_feature_df.columns)
    coverage_sum = np.zeros(len(feature_names), dtype=np.float64)

    for _, row in mixtures_train.iterrows():
        comp = row["Компонент"]
        party = row["Наименование партии"]
        key = (comp, party)
        if key in X_mask_df.index:
            mask = X_mask_df.loc[key].values.astype(np.float32)
        else:
            mask = typical_masks_map.get(comp, np.zeros(len(feature_names), dtype=np.float32))
        coverage_sum += mask

    coverage = coverage_sum / max(len(mixtures_train), 1)
    coverage_series = pd.Series(coverage, index=feature_names).sort_values(ascending=False)
    kept_columns = coverage_series[coverage_series >= min_coverage].index.tolist()

    if not kept_columns:
        kept_columns = coverage_series.head(min(10, len(coverage_series))).index.tolist()

    filtered_feature_df = X_feature_df[kept_columns].copy()
    filtered_mask_df = X_mask_df[kept_columns].copy()

    filtered_typical_features_map = {
        comp: values[[feature_names.index(col) for col in kept_columns]].astype(np.float32)
        for comp, values in typical_features_map.items()
    }
    filtered_typical_masks_map = {
        comp: values[[feature_names.index(col) for col in kept_columns]].astype(np.float32)
        for comp, values in typical_masks_map.items()
    }

    return (
        filtered_feature_df,
        filtered_mask_df,
        filtered_typical_features_map,
        filtered_typical_masks_map,
        kept_columns,
        coverage_series,
    )


def filter_categorical_features_by_train_coverage(
    mixtures_train: pd.DataFrame,
    categorical_feature_df: pd.DataFrame,
    typical_categorical_map: dict[str, np.ndarray],
    min_coverage: float = 0.08,
    max_cardinality: int = 20,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str], pd.Series, pd.Series]:
    feature_names = list(categorical_feature_df.columns)
    if not feature_names:
        empty_series = pd.Series(dtype=float)
        return categorical_feature_df, typical_categorical_map, [], empty_series, empty_series

    coverage_sum = np.zeros(len(feature_names), dtype=np.float64)
    observed_values: dict[str, set[str]] = {name: set() for name in feature_names}

    for _, row in mixtures_train.iterrows():
        comp = row["Компонент"]
        party = row["Наименование партии"]
        key = (comp, party)
        if key in categorical_feature_df.index:
            values = categorical_feature_df.loc[key].values.astype(object)
        else:
            values = typical_categorical_map.get(comp, np.array(["__MISSING__"] * len(feature_names), dtype=object))

        for idx, value in enumerate(values):
            value_str = str(value)
            if value_str != "__MISSING__":
                coverage_sum[idx] += 1.0
                observed_values[feature_names[idx]].add(value_str)

    coverage = coverage_sum / max(len(mixtures_train), 1)
    coverage_series = pd.Series(coverage, index=feature_names).sort_values(ascending=False)
    cardinality_series = pd.Series(
        {feature_name: len(values) for feature_name, values in observed_values.items()}
    ).sort_values(ascending=False)

    kept_columns = [
        column
        for column in feature_names
        if coverage_series.get(column, 0.0) >= min_coverage
        and cardinality_series.get(column, 0) <= max_cardinality
    ]

    filtered_categorical_df = categorical_feature_df[kept_columns].copy() if kept_columns else pd.DataFrame(index=categorical_feature_df.index)
    original_positions = {name: idx for idx, name in enumerate(feature_names)}
    filtered_typical_categorical_map = {
        comp: values[[original_positions[col] for col in kept_columns]].astype(object)
        for comp, values in typical_categorical_map.items()
    }

    return (
        filtered_categorical_df,
        filtered_typical_categorical_map,
        kept_columns,
        coverage_series,
        cardinality_series,
    )


def build_categorical_encoders(
    categorical_feature_df: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    encoders: dict[str, dict[str, int]] = {}
    for column in categorical_feature_df.columns:
        values = sorted(set(categorical_feature_df[column].astype(str).tolist()) | {"__MISSING__"})
        encoders[column] = {value: idx + 1 for idx, value in enumerate(values)}
    return encoders


def fit_scalers(mixtures_train: pd.DataFrame) -> tuple[StandardScaler, StandardScaler, StandardScaler, LabelEncoder, StandardScaler, StandardScaler]:
    temp_vals = mixtures_train["Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"].values.reshape(-1, 1)
    time_vals = mixtures_train["Время испытания | - Daimler Oxidation Test (DOT), ч"].values.reshape(-1, 1)
    biofuel_vals = mixtures_train["Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"].values.reshape(-1, 1)
    catalyst_raw = mixtures_train["Дозировка катализатора, категория"].astype(str).values

    scaler_temp = StandardScaler().fit(temp_vals)
    scaler_time = StandardScaler().fit(time_vals)
    scaler_biofuel = StandardScaler().fit(biofuel_vals)

    le_catalyst = LabelEncoder().fit(catalyst_raw)
    catalyst_encoded = le_catalyst.transform(catalyst_raw).reshape(-1, 1)
    scaler_catalyst = StandardScaler().fit(catalyst_encoded)

    targets = (
        mixtures_train.groupby("scenario_id")[TRAIN_TARGETS]
        .first()
        .reset_index(drop=True)
        .values
    )
    scaler_y = StandardScaler().fit(targets)
    return scaler_temp, scaler_time, scaler_biofuel, le_catalyst, scaler_catalyst, scaler_y


def get_condition_features(
    scenario_df: pd.DataFrame,
    scaler_temp: StandardScaler,
    scaler_time: StandardScaler,
    scaler_biofuel: StandardScaler,
    le_catalyst: LabelEncoder,
    scaler_catalyst: StandardScaler,
) -> np.ndarray:
    first = scenario_df.iloc[0]
    temp = scaler_temp.transform([[first["Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"]]])[0, 0]
    time_value = scaler_time.transform([[first["Время испытания | - Daimler Oxidation Test (DOT), ч"]]])[0, 0]
    bio = scaler_biofuel.transform([[first["Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"]]])[0, 0]
    catalyst_code = le_catalyst.transform([str(first["Дозировка катализатора, категория"])])[0]
    catalyst = scaler_catalyst.transform([[catalyst_code]])[0, 0]
    return np.array([temp, time_value, bio, catalyst], dtype=np.float32)


class ScenarioSetDataset(Dataset):
    def __init__(
        self,
        mixtures: pd.DataFrame,
        scenario_ids: list[str],
        X_feature_df: pd.DataFrame,
        X_mask_df: pd.DataFrame,
        typical_features_map: dict[str, np.ndarray],
        typical_masks_map: dict[str, np.ndarray],
        categorical_feature_df: pd.DataFrame,
        typical_categorical_map: dict[str, np.ndarray],
        categorical_encoders: dict[str, dict[str, int]],
        scaler_temp: StandardScaler,
        scaler_time: StandardScaler,
        scaler_biofuel: StandardScaler,
        le_catalyst: LabelEncoder,
        scaler_catalyst: StandardScaler,
        scaler_y: StandardScaler | None,
        target_index: int | None = None,
    ) -> None:
        self.mixtures = mixtures
        self.scenario_ids = list(scenario_ids)
        self.X_feature_df = X_feature_df
        self.X_mask_df = X_mask_df
        self.typical_features_map = typical_features_map
        self.typical_masks_map = typical_masks_map
        self.categorical_feature_df = categorical_feature_df
        self.typical_categorical_map = typical_categorical_map
        self.categorical_encoders = categorical_encoders
        self.categorical_feature_names = list(categorical_feature_df.columns)
        self.scaler_temp = scaler_temp
        self.scaler_time = scaler_time
        self.scaler_biofuel = scaler_biofuel
        self.le_catalyst = le_catalyst
        self.scaler_catalyst = scaler_catalyst
        self.scaler_y = scaler_y
        self.target_index = target_index

    def __len__(self) -> int:
        return len(self.scenario_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        scenario_id = self.scenario_ids[idx]
        scenario_df = self.mixtures[self.mixtures["scenario_id"] == scenario_id]

        component_values = []
        feature_masks = []
        categorical_ids = []
        mass_fractions = []

        for _, row in scenario_df.iterrows():
            comp = row["Компонент"]
            party = row["Наименование партии"]
            key = (comp, party)

            if key in self.X_feature_df.index:
                feat = self.X_feature_df.loc[key].values.astype(np.float32)
                mask = self.X_mask_df.loc[key].values.astype(np.float32)
            else:
                feat = self.typical_features_map.get(comp)
                mask = self.typical_masks_map.get(comp)

            if feat is None or mask is None:
                feat = np.zeros(self.X_feature_df.shape[1], dtype=np.float32)
                mask = np.zeros(self.X_feature_df.shape[1], dtype=np.float32)

            if self.categorical_feature_names:
                if key in self.categorical_feature_df.index:
                    cat_values = self.categorical_feature_df.loc[key].values.astype(object)
                else:
                    cat_values = self.typical_categorical_map.get(
                        comp,
                        np.array(["__MISSING__"] * len(self.categorical_feature_names), dtype=object),
                    )
                cat_ids = [
                    self.categorical_encoders[column].get(str(value), self.categorical_encoders[column]["__MISSING__"])
                    for column, value in zip(self.categorical_feature_names, cat_values)
                ]
            else:
                cat_ids = []

            component_values.append(feat)
            feature_masks.append(mask)
            categorical_ids.append(cat_ids)
            mass_fractions.append(float(row["Массовая доля, %"]) / 100.0)

        component_values_arr = np.stack(component_values).astype(np.float32)
        feature_masks_arr = np.stack(feature_masks).astype(np.float32)
        categorical_ids_arr = (
            np.asarray(categorical_ids, dtype=np.int64)
            if self.categorical_feature_names
            else np.zeros((len(component_values), 0), dtype=np.int64)
        )

        mass_fractions_arr = np.asarray(mass_fractions, dtype=np.float32)
        mass_sum = mass_fractions_arr.sum()
        if mass_sum > 0:
            mass_fractions_arr = mass_fractions_arr / mass_sum
        else:
            mass_fractions_arr = np.full_like(mass_fractions_arr, 1.0 / len(mass_fractions_arr))

        item: dict[str, torch.Tensor | str] = {
            "scenario_id": scenario_id,
            "component_values": torch.tensor(component_values_arr, dtype=torch.float32),
            "feature_masks": torch.tensor(feature_masks_arr, dtype=torch.float32),
            "component_cat_ids": torch.tensor(categorical_ids_arr, dtype=torch.long),
            "mass_fractions": torch.tensor(mass_fractions_arr, dtype=torch.float32),
            "conditions": torch.tensor(
                get_condition_features(
                    scenario_df,
                    self.scaler_temp,
                    self.scaler_time,
                    self.scaler_biofuel,
                    self.le_catalyst,
                    self.scaler_catalyst,
                ),
                dtype=torch.float32,
            ),
        }

        if self.scaler_y is not None and self.target_index is not None:
            target_value = float(scenario_df.iloc[0][TRAIN_TARGETS[self.target_index]])
            target_norm = (target_value - float(self.scaler_y.mean_[self.target_index])) / float(self.scaler_y.scale_[self.target_index])
            item["targets"] = torch.tensor([target_norm], dtype=torch.float32)

        return item


def collate_set_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    batch_size = len(batch)
    max_components = max(int(item["component_values"].shape[0]) for item in batch)
    num_features = int(batch[0]["component_values"].shape[1])
    num_categorical = int(batch[0]["component_cat_ids"].shape[1])

    component_values = torch.zeros(batch_size, max_components, num_features, dtype=torch.float32)
    feature_masks = torch.zeros(batch_size, max_components, num_features, dtype=torch.float32)
    component_cat_ids = torch.zeros(batch_size, max_components, num_categorical, dtype=torch.long)
    mass_fractions = torch.zeros(batch_size, max_components, dtype=torch.float32)
    component_mask = torch.zeros(batch_size, max_components, dtype=torch.float32)
    conditions = torch.zeros(batch_size, 4, dtype=torch.float32)

    targets = None
    if "targets" in batch[0]:
        target_dim = int(batch[0]["targets"].numel())
        targets = torch.zeros(batch_size, target_dim, dtype=torch.float32)

    scenario_ids: list[str] = []
    for row_idx, item in enumerate(batch):
        length = int(item["component_values"].shape[0])
        scenario_ids.append(str(item["scenario_id"]))
        component_values[row_idx, :length] = item["component_values"]
        feature_masks[row_idx, :length] = item["feature_masks"]
        component_cat_ids[row_idx, :length] = item["component_cat_ids"]
        mass_fractions[row_idx, :length] = item["mass_fractions"]
        component_mask[row_idx, :length] = 1.0
        conditions[row_idx] = item["conditions"]
        if targets is not None:
            targets[row_idx] = item["targets"]

    result: dict[str, torch.Tensor | list[str]] = {
        "scenario_ids": scenario_ids,
        "component_values": component_values,
        "feature_masks": feature_masks,
        "component_cat_ids": component_cat_ids,
        "mass_fractions": mass_fractions,
        "component_mask": component_mask,
        "conditions": conditions,
    }
    if targets is not None:
        result["targets"] = targets
    return result


class ComponentEncoder(nn.Module):
    def __init__(self, n_component_features: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_component_features = n_component_features
        pairwise_dim = n_component_features * n_component_features
        raw_summary_dim = n_component_features * 2 + 1 + 4

        self.base_projection = nn.Sequential(
            nn.Linear(pairwise_dim * 2 + n_component_features + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.hypernet = nn.Sequential(
            nn.Linear(raw_summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        component_values: torch.Tensor,
        feature_masks: torch.Tensor,
        mass_fractions: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        masked_values = component_values * feature_masks
        pairwise_values = torch.einsum("bif, big -> bifg", masked_values, masked_values).reshape(
            component_values.size(0), component_values.size(1), -1
        )
        pairwise_masks = torch.einsum("bif, big -> bifg", feature_masks, feature_masks).reshape(
            feature_masks.size(0), feature_masks.size(1), -1
        )

        base_input = torch.cat(
            [pairwise_values, pairwise_masks, masked_values, mass_fractions.unsqueeze(-1)],
            dim=-1,
        )
        base_embedding = self.base_projection(base_input)

        expanded_conditions = conditions.unsqueeze(1).expand(-1, component_values.size(1), -1)
        hyper_input = torch.cat(
            [component_values, feature_masks, mass_fractions.unsqueeze(-1), expanded_conditions],
            dim=-1,
        )
        gamma_beta = self.hypernet(hyper_input)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        modulated = base_embedding * (1.0 + torch.tanh(gamma)) + beta
        return self.post(modulated)


class MAB(nn.Module):
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln: bool = False) -> None:
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), dim=2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), dim=0), dim=2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False) -> None:
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, ln: bool = False) -> None:
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int, ln: bool = False) -> None:
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input: int,
        num_outputs: int,
        dim_output: int,
        num_inds: int = 32,
        dim_hidden: int = 128,
        num_heads: int = 4,
        ln: bool = False,
    ) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(X))


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, encoded_components: torch.Tensor, component_mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(encoded_components).squeeze(-1)
        logits = logits.masked_fill(component_mask == 0, -1e9)
        weights = torch.softmax(logits, dim=1)
        weights = weights * component_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(encoded_components * weights.unsqueeze(-1), dim=1)


def masked_mean_pool(encoded_components: torch.Tensor, component_mask: torch.Tensor) -> torch.Tensor:
    weights = component_mask.unsqueeze(-1)
    return (encoded_components * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_max_pool(encoded_components: torch.Tensor, component_mask: torch.Tensor) -> torch.Tensor:
    masked = encoded_components.masked_fill(component_mask.unsqueeze(-1) == 0, -1e9)
    return masked.max(dim=1).values


class DeepSetSumRegressor(nn.Module):
    def __init__(
        self,
        n_component_features: int,
        categorical_cardinalities: list[int] | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        categorical_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.component_encoder = ComponentEncoder(n_component_features, hidden_dim=hidden_dim, dropout=dropout)
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, categorical_embedding_dim, padding_idx=0) for cardinality in (categorical_cardinalities or [])]
        )
        self.categorical_projection = (
            nn.Sequential(
                nn.Linear(categorical_embedding_dim * len(self.categorical_embeddings), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            if self.categorical_embeddings
            else None
        )
        self.attention_pool = AttentionPooling(hidden_dim)
        self.condition_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        component_values: torch.Tensor,
        feature_masks: torch.Tensor,
        component_cat_ids: torch.Tensor,
        mass_fractions: torch.Tensor,
        component_mask: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        encoded_components = self.component_encoder(component_values, feature_masks, mass_fractions, conditions)
        if self.categorical_projection is not None:
            categorical_embeds = [
                embedding(component_cat_ids[:, :, idx])
                for idx, embedding in enumerate(self.categorical_embeddings)
            ]
            categorical_representation = self.categorical_projection(torch.cat(categorical_embeds, dim=-1))
            encoded_components = encoded_components + categorical_representation
        encoded_components = encoded_components * component_mask.unsqueeze(-1)
        mean_representation = masked_mean_pool(encoded_components, component_mask)
        max_representation = masked_max_pool(encoded_components, component_mask)
        attention_representation = self.attention_pool(encoded_components, component_mask)
        condition_representation = self.condition_net(conditions)
        joint = torch.cat([mean_representation, max_representation, attention_representation, condition_representation], dim=-1)
        return self.head(joint)


def move_batch_to_device(batch: dict[str, torch.Tensor | list[str]], device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau,
    epochs: int = 120,
    device: str = "cpu",
    early_stopping_patience: int | None = 20,
) -> tuple[list[float], list[float], dict[str, object]]:
    model.to(device)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_snapshot = {
        "epoch": 0,
        "val_loss": float("inf"),
        "model": deepcopy(model).cpu(),
        "state_dict": deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()}),
    }
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = move_batch_to_device(batch, torch.device(device))
            optimizer.zero_grad()
            outputs = model(
                batch["component_values"],
                batch["feature_masks"],
                batch["component_cat_ids"],
                batch["mass_fractions"],
                batch["component_mask"],
                batch["conditions"],
            )
            loss = criterion(outputs, batch["targets"])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, torch.device(device))
                outputs = model(
                    batch["component_values"],
                    batch["feature_masks"],
                    batch["component_cat_ids"],
                    batch["mass_fractions"],
                    batch["component_mask"],
                    batch["conditions"],
                )
                loss = criterion(outputs, batch["targets"])
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if early_stopping_patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_snapshot = {
                    "epoch": epoch + 1,
                    "val_loss": avg_val_loss,
                    "model": deepcopy(model).cpu(),
                    "state_dict": deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()}),
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

    model.load_state_dict(best_snapshot["state_dict"])
    model.to(device)
    model.best_snapshot = best_snapshot
    print(
        f"Restored best model from epoch {best_snapshot['epoch']} "
        f"with Val Loss={best_snapshot['val_loss']:.6f}"
    )
    return train_losses, val_losses, best_snapshot


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler_y: StandardScaler,
    target_index: int,
) -> dict[str, np.ndarray | float]:
    model.eval()
    preds_norm: list[np.ndarray] = []
    targets_norm: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                batch["component_values"],
                batch["feature_masks"],
                batch["component_cat_ids"],
                batch["mass_fractions"],
                batch["component_mask"],
                batch["conditions"],
            )
            preds_norm.append(outputs.cpu().numpy())
            targets_norm.append(batch["targets"].cpu().numpy())

    preds_norm_arr = np.concatenate(preds_norm, axis=0)
    targets_norm_arr = np.concatenate(targets_norm, axis=0)
    preds = preds_norm_arr[:, 0] * float(scaler_y.scale_[target_index]) + float(scaler_y.mean_[target_index])
    targets = targets_norm_arr[:, 0] * float(scaler_y.scale_[target_index]) + float(scaler_y.mean_[target_index])

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "predictions": preds,
        "targets": targets,
    }


def predict_on_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler_y: StandardScaler,
    target_index: int,
) -> np.ndarray:
    model.eval()
    predictions_norm: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                batch["component_values"],
                batch["feature_masks"],
                batch["component_cat_ids"],
                batch["mass_fractions"],
                batch["component_mask"],
                batch["conditions"],
            )
            predictions_norm.append(outputs.cpu().numpy())

    predictions_norm_arr = np.concatenate(predictions_norm, axis=0)
    return predictions_norm_arr[:, 0] * float(scaler_y.scale_[target_index]) + float(scaler_y.mean_[target_index])


def run_pipeline() -> None:
    set_global_seed(RANDOM_SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
        feature_coverage,
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
        categorical_coverage,
        categorical_cardinality,
    ) = filter_categorical_features_by_train_coverage(
        mixtures_train=mixtures_train,
        categorical_feature_df=categorical_feature_df,
        typical_categorical_map=typical_categorical_map,
        min_coverage=0.08,
        max_cardinality=20,
    )
    categorical_encoders = build_categorical_encoders(categorical_feature_df)
    scaler_temp, scaler_time, scaler_biofuel, le_catalyst, scaler_catalyst, scaler_y = fit_scalers(mixtures_train)

    print(f"Kept {len(component_feature_names)} numeric component features after coverage filter")
    print(feature_coverage.head(10).to_string())
    print(f"Kept {len(categorical_feature_names)} categorical component features after filters")
    if categorical_feature_names:
        print("Categorical features:", categorical_feature_names)
        print(categorical_coverage[categorical_feature_names].to_string())
        print(categorical_cardinality[categorical_feature_names].to_string())

    train_scenario_ids = mixtures_train["scenario_id"].unique().tolist()
    train_ids, val_ids = train_test_split(train_scenario_ids, test_size=0.2, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    test_scenario_ids = mixtures_test["scenario_id"].unique().tolist()
    test_predictions_by_target: dict[int, np.ndarray] = {}
    feature_state = {
        "component_feature_names": component_feature_names,
        "categorical_feature_names": categorical_feature_names,
        "categorical_encoders": categorical_encoders,
        "typical_features_map": typical_features_map,
        "typical_masks_map": typical_masks_map,
        "typical_categorical_map": typical_categorical_map,
        "scaler_temp_mean": scaler_temp.mean_.copy(),
        "scaler_temp_scale": scaler_temp.scale_.copy(),
        "scaler_time_mean": scaler_time.mean_.copy(),
        "scaler_time_scale": scaler_time.scale_.copy(),
        "scaler_biofuel_mean": scaler_biofuel.mean_.copy(),
        "scaler_biofuel_scale": scaler_biofuel.scale_.copy(),
        "scaler_catalyst_mean": scaler_catalyst.mean_.copy(),
        "scaler_catalyst_scale": scaler_catalyst.scale_.copy(),
        "label_encoder_catalyst_classes": le_catalyst.classes_.copy(),
        "scaler_y_mean": scaler_y.mean_.copy(),
        "scaler_y_scale": scaler_y.scale_.copy(),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "seed": RANDOM_SEED,
    }

    for target_index, target_name in enumerate(TRAIN_TARGETS):
        print(f"\nTraining single-target model for: {target_name}")

        train_dataset = ScenarioSetDataset(
            mixtures=mixtures_train,
            scenario_ids=train_ids,
            X_feature_df=X_feature_df,
            X_mask_df=X_mask_df,
            typical_features_map=typical_features_map,
            typical_masks_map=typical_masks_map,
            categorical_feature_df=categorical_feature_df,
            typical_categorical_map=typical_categorical_map,
            categorical_encoders=categorical_encoders,
            scaler_temp=scaler_temp,
            scaler_time=scaler_time,
            scaler_biofuel=scaler_biofuel,
            le_catalyst=le_catalyst,
            scaler_catalyst=scaler_catalyst,
            scaler_y=scaler_y,
            target_index=target_index,
        )
        val_dataset = ScenarioSetDataset(
            mixtures=mixtures_train,
            scenario_ids=val_ids,
            X_feature_df=X_feature_df,
            X_mask_df=X_mask_df,
            typical_features_map=typical_features_map,
            typical_masks_map=typical_masks_map,
            categorical_feature_df=categorical_feature_df,
            typical_categorical_map=typical_categorical_map,
            categorical_encoders=categorical_encoders,
            scaler_temp=scaler_temp,
            scaler_time=scaler_time,
            scaler_biofuel=scaler_biofuel,
            le_catalyst=le_catalyst,
            scaler_catalyst=scaler_catalyst,
            scaler_y=scaler_y,
            target_index=target_index,
        )
        full_train_dataset = ScenarioSetDataset(
            mixtures=mixtures_train,
            scenario_ids=train_scenario_ids,
            X_feature_df=X_feature_df,
            X_mask_df=X_mask_df,
            typical_features_map=typical_features_map,
            typical_masks_map=typical_masks_map,
            categorical_feature_df=categorical_feature_df,
            typical_categorical_map=typical_categorical_map,
            categorical_encoders=categorical_encoders,
            scaler_temp=scaler_temp,
            scaler_time=scaler_time,
            scaler_biofuel=scaler_biofuel,
            le_catalyst=le_catalyst,
            scaler_catalyst=scaler_catalyst,
            scaler_y=scaler_y,
            target_index=target_index,
        )
        test_dataset = ScenarioSetDataset(
            mixtures=mixtures_test,
            scenario_ids=test_scenario_ids,
            X_feature_df=X_feature_df,
            X_mask_df=X_mask_df,
            typical_features_map=typical_features_map,
            typical_masks_map=typical_masks_map,
            categorical_feature_df=categorical_feature_df,
            typical_categorical_map=typical_categorical_map,
            categorical_encoders=categorical_encoders,
            scaler_temp=scaler_temp,
            scaler_time=scaler_time,
            scaler_biofuel=scaler_biofuel,
            le_catalyst=le_catalyst,
            scaler_catalyst=scaler_catalyst,
            scaler_y=None,
            target_index=None,
        )

        train_generator = torch.Generator().manual_seed(RANDOM_SEED + target_index)
        full_train_generator = torch.Generator().manual_seed(RANDOM_SEED + 100 + target_index)
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_set_batch,
            generator=train_generator,
        )
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_set_batch)
        full_train_loader = DataLoader(
            full_train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_set_batch,
            generator=full_train_generator,
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_set_batch)

        model_kwargs = {
            "n_component_features": len(component_feature_names),
            "categorical_cardinalities": [len(categorical_encoders[column]) for column in categorical_feature_names],
            "hidden_dim": 256,
            "dropout": 0.15,
        }

        model = DeepSetSumRegressor(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=140,
            device=str(device),
            early_stopping_patience=25,
        )

        metrics = evaluate_model(model, val_loader, device, scaler_y, target_index=target_index)
        print(f"Validation MAE: {metrics['mae']:.4f}")
        print(f"Validation RMSE: {metrics['rmse']:.4f}")

        final_model = DeepSetSumRegressor(**model_kwargs).to(device)
        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-5)
        final_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode="min", patience=10)

        _, _, best_snapshot = train_model(
            model=final_model,
            train_loader=full_train_loader,
            val_loader=val_loader,
            optimizer=final_optimizer,
            criterion=criterion,
            scheduler=final_scheduler,
            epochs=140,
            device=str(device),
            early_stopping_patience=25,
        )

        test_predictions_by_target[target_index] = predict_on_test(final_model, test_loader, device, scaler_y, target_index=target_index)
        target_slug = make_target_slug(target_name)
        save_model_artifact(
            save_path=ARTIFACTS_DIR / f"deepset_sum_{target_slug}.pt",
            model=final_model.cpu(),
            model_kwargs=model_kwargs,
            target_name=target_name,
            target_index=target_index,
            feature_state=feature_state,
            training_state={
                "optimizer_class": "Adam",
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
                "scheduler_class": "ReduceLROnPlateau",
                "scheduler_patience": 10,
                "epochs": 140,
                "early_stopping_patience": 25,
                "batch_size": 16,
                "device": str(device),
                "best_epoch": best_snapshot["epoch"],
                "best_val_loss": best_snapshot["val_loss"],
            },
        )
        final_model.to(device)
        print(f"Saved model artifact: {ARTIFACTS_DIR / f'deepset_sum_{target_slug}.pt'}")

    results_df = pd.DataFrame(
        {
            "scenario_id": test_scenario_ids,
            TRAIN_TARGETS[0]: test_predictions_by_target[0],
            TRAIN_TARGETS[1]: test_predictions_by_target[1],
        }
    )
    results_df.to_csv(PROJECT_DIR / "test_predictions_deepset_sum.csv", index=False)
    print("Saved:", PROJECT_DIR / "test_predictions_deepset_sum.csv")


if __name__ == "__main__":
    run_pipeline()
