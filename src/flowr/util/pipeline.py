"""Pipeline utility functions."""
from typing import Dict, List

import torch


def average_metrics_list(
    metrics_dict_list: List[Dict[str, torch.Tensor]], get_std: bool = True
) -> Dict[str, torch.Tensor]:
    """Given a list of of dictionaries with metrics, return avg and optionally stddev."""
    metrics_dict = {}
    for key in metrics_dict_list[0].keys():
        metric_values = torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
        metric_values = metric_values[~torch.isnan(metric_values)]
        if get_std:
            key_std, key_mean = torch.std_mean(metric_values)
            metrics_dict[key] = float(key_mean)
            metrics_dict[f"{key}_std"] = float(key_std)
        else:
            metrics_dict[key] = float(metric_values.mean())
    return metrics_dict
