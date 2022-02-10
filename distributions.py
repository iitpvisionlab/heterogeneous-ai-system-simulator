import math
import random

import numpy as np


class NonNegativeUniformDistribution:
    def __init__(self, mean: float, std: float):
        # D = ((b - a)^2) / 12
        width = std * math.sqrt(12)
        self.min = max(0, mean - width / 2)
        self.max = mean + width / 2

    def __call__(self):
        return random.uniform(self.min, self.max)


class EmpiricalDistribution:
    def __init__(self, samples: list):
        self.samples = samples

    def __call__(self):
        return random.choice(self.samples)


# For testing purposes
uniform_exec_time_distributions = {
    "decoder": NonNegativeUniformDistribution(6.10, 0.13),
    "nn_inputs_preparator": NonNegativeUniformDistribution(9.76, 3.36),
    "nn_inferencer": NonNegativeUniformDistribution(16.56, 3.0),
    "nn_postprocessor": NonNegativeUniformDistribution(9.10, 3.27),
    "tracker": NonNegativeUniformDistribution(0.05, 0.01),
    "associator": NonNegativeUniformDistribution(0.01, 0.0),
    "frame_to_world": NonNegativeUniformDistribution(0.02, 0.0),
    "geometry_filter": NonNegativeUniformDistribution(0.001, 0.001),
    "multicamera_tracker": NonNegativeUniformDistribution(0.01, 0),
    "tag_integrator": NonNegativeUniformDistribution(0.001, 0.001),
    "detector_result_join": NonNegativeUniformDistribution(0.02, 0),
}


def make_empirical_distributions_from_traces(traces: dict, use_uniform: bool = False):
    distr_store = {}

    for node_name, samples in traces.items():
        if node_name.startswith("tbb_"):
            continue
        samples = [sample / 1e6 for sample in samples]  # convert to ms

        if use_uniform:
            distr_store[node_name] = NonNegativeUniformDistribution(
                np.mean(samples), np.std(samples)
            )
        else:
            distr_store[node_name] = EmpiricalDistribution(samples)

    return distr_store
