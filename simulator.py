from enum import Enum, auto

import simpy


class NNCacheType(Enum):
    NOT_ACTIVATED = auto()
    REAL_FULL = auto()
    IDEAL_FULL = auto()

    def __str__(self):
        return self.name


class Simulator:
    def __init__(
        self,
        cpu_core_count: int,
        distributions: dict,
        nn_cache_type: NNCacheType,
        new_module_enabled: bool,
        use_async: bool
    ):
        self.env = simpy.Environment()
        self.cpu = simpy.Resource(self.env, capacity=cpu_core_count)
        self.gpu = simpy.Resource(self.env, capacity=1)
        self.mt_mutex = simpy.Resource(self.env, capacity=1)
        self.distributions = distributions
        self.nn_compiled = False
        self.nn_cache_type = nn_cache_type
        self.nn_cache_mutex = simpy.Resource(self.env, capacity=1)
        self.new_module_enabled = new_module_enabled
        self.use_async = use_async

    def run(self):
        self.env.run()

    def now(self):
        return self.env.now
