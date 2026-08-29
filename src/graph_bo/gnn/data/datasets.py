from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch_geometric.data import Data

from graph_bo.gnn.data.tensor_builder import ADSGTensorBuilder


def load_csv_graphs(
    csv_path: Path,
    tensor_builder: ADSGTensorBuilder,
    target_columns: Sequence[str] = (),
) -> list[Data]:
    df = pd.read_csv(csv_path).sort_values("sample_id").reset_index(drop=True)
    x_columns = [f"x{i}" for i in range(len(tensor_builder.graph_processor.des_vars))]
    x_values = df.loc[:, x_columns].to_numpy(dtype=float)

    if target_columns:
        target_values = df.loc[:, list(target_columns)].to_numpy(dtype=float)

    data_list: list[Data] = []
    for i, row in df.iterrows():
        data = tensor_builder.from_vector(x_values[i])
        data.sample_id = int(row["sample_id"])
        if target_columns:
            data.graph_targets = torch.tensor(target_values[i], dtype=torch.float32)
        data_list.append(data)

    return data_list


def split_graphs_by_sample_id(
    data_list: Sequence[Data],
    n_folds: int,
    val_fold: int,
) -> tuple[list[Data], list[Data]]:
    train_graphs = [data for data in data_list if data.sample_id % n_folds != val_fold]
    val_graphs = [data for data in data_list if data.sample_id % n_folds == val_fold]
    return train_graphs, val_graphs


def split_graphs_train_val_test_by_sample_id(
    data_list: Sequence[Data],
    n_folds: int,
    val_fold: int,
    test_fold: int,
) -> tuple[list[Data], list[Data], list[Data]]:
    if n_folds < 3:
        raise ValueError("n_folds must be at least 3 for separate train/validation/test splits.")
    if val_fold == test_fold:
        raise ValueError(f"val_fold and test_fold must differ, got {val_fold}.")

    train_graphs = [
        data
        for data in data_list
        if data.sample_id % n_folds not in {val_fold, test_fold}
    ]
    val_graphs = [data for data in data_list if data.sample_id % n_folds == val_fold]
    test_graphs = [data for data in data_list if data.sample_id % n_folds == test_fold]
    return train_graphs, val_graphs, test_graphs
