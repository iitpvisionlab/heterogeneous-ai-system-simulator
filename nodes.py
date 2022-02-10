from simpy import Process, Resource, Store

from simulator import NNCacheType, Simulator


class BaseNode:
    def __init__(self, simulator: Simulator, name: str):
        self.simulator = simulator
        self.name = name
        self.distribution = self.simulator.distributions[name]
        self.successors = []
        self.task_queue = Store(self.simulator.env)

    def add_successor(self, node: "BaseNode"):
        self.successors.append(node)

    def process_tasks(self) -> Process:
        self.simulator.env.process(self._run())

    def _run(self):
        raise NotImplementedError()


class FunctionNode(BaseNode):
    def _run(self):
        while True:
            task = yield self.task_queue.get()
            with self.simulator.cpu.request() as request:
                yield request

                yield self.simulator.env.timeout(self.distribution())

                for successor in self.successors:
                    successor.task_queue.put(task)

                if len(self.successors) == 0:
                    task.put(None)


class GPUNode(BaseNode):
    def __init__(self, simulator: Simulator, name: str, compile_time_distr):
        super().__init__(simulator, name)
        self.compile_time_distr = compile_time_distr

    def _run(self):
        while True:
            task = yield self.task_queue.get()
            with self.simulator.cpu.request() as request:
                yield request

            with self.simulator.gpu.request() as gpu_request:
                yield gpu_request

                if not self.simulator.nn_compiled:
                    yield self.simulator.env.timeout(self.compile_time_distr())
                    self.simulator.nn_compiled = True
                else:
                    yield self.simulator.env.timeout(self.distribution())

                for successor in self.successors:
                    successor.task_queue.put(task)

                if len(self.successors) == 0:
                    task.put(None)


class MutexFunctionNode(BaseNode):
    def __init__(self, simulator: Simulator, name: str, mutex: Resource):
        super().__init__(simulator, name)
        self.mutex = mutex

    def _run(self):
        while True:
            task = yield self.task_queue.get()
            with self.simulator.cpu.request() as request:
                yield request

                with self.mutex.request() as mutex_request:
                    yield mutex_request

                    yield self.simulator.env.timeout(self.distribution())

                    for successor in self.successors:
                        successor.task_queue.put(task)

                    if len(self.successors) == 0:
                        task.put(None)


class BufferNode(BaseNode):
    def __init__(self, simulator: Simulator, name: str, buffer_size: int):
        super().__init__(simulator, name)
        self.buffer_size = buffer_size
        self.current_buffer_size = 0

    def _run(self):
        while True:
            task = yield self.task_queue.get()
            with self.simulator.cpu.request() as request:
                yield request

                self.current_buffer_size += 1
                if self.current_buffer_size == self.buffer_size:
                    yield self.simulator.env.timeout(self.distribution())

                    for successor in self.successors:
                        successor.task_queue.put(task)

                    if len(self.successors) == 0:
                        task.put(None)

                    self.current_buffer_size = 0


class MultiFunctionNode(BaseNode):
    def __init__(self, simulator: Simulator, name: str, output_ports: int):
        super().__init__(simulator, name)
        self.successors = [[] for _ in range(output_ports)]

    def add_successor(self, port: int, node: "BaseNode"):
        self.successors[port].append(node)


class NNCacherSearcherNode(MultiFunctionNode):
    def __init__(self, simulator: Simulator, name: str):
        super().__init__(simulator, name, 2)

    def _run(self):
        while True:
            task = yield self.task_queue.get()
            with self.simulator.cpu.request() as request:
                yield request

                with self.simulator.nn_cache_mutex.request() as mutex_request:
                    yield mutex_request

                    yield self.simulator.env.timeout(self.distribution())

                    output_port = (
                        0
                        if self.simulator.nn_cache_type is NNCacheType.NOT_ACTIVATED
                        else 1
                    )
                    successors = self.successors[output_port]

                    for successor in successors:
                        successor.task_queue.put(task)

                    if len(successors) == 0:
                        task.put(None)


class FrameSelectorNode(MultiFunctionNode):
    def __init__(self, simulator: Simulator, name: str, stride: int):
        super().__init__(simulator, name, 2)
        self.stride = stride
        self.frame_idx = 0

    def _run(self):
        while True:
            task = yield self.task_queue.get()
            self.frame_idx += 1

            with self.simulator.cpu.request() as request:
                yield request

                yield self.simulator.env.timeout(self.distribution())

                output_port = 0 if self.frame_idx % self.stride == 0 else 1
                successors = self.successors[output_port]

                for successor in successors:
                    successor.task_queue.put(task)

                if len(successors) == 0:
                    task.put(None)
