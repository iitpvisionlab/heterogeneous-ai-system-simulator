from imp import new_module
from simpy import Store
from nodes import (
    FrameSelectorNode,
    FunctionNode,
    GPUNode,
    MutexFunctionNode,
    BufferNode,
    NNCacherSearcherNode,
)
from simulator import Simulator


class VideoStreamFlowGraph:
    def __init__(self, simulator: Simulator) -> None:
        self.simulator = simulator

        self.nodes = {
            "decoder": FunctionNode(simulator, "decoder"),
            "frame_corrector": FunctionNode(simulator, "frame_to_detector"),
            "frame_to_detector": FunctionNode(simulator, "frame_to_detector"),
            "frame_selector": FrameSelectorNode(simulator, "frame_selector", 1),
            "nn_cache_searcher": NNCacherSearcherNode(simulator, "nn_cache_searcher"),
            "nn_inputs_preparator": FunctionNode(simulator, "nn_inputs_preparator"),
            "nn_inferencer": GPUNode(
                simulator, "nn_inferencer", simulator.distributions["nn_compilation"]
            ),
            "nn_postprocessor": FunctionNode(simulator, "nn_postprocessor"),
            "nn_cache_writer": MutexFunctionNode(
                simulator, "nn_cache_writer", simulator.nn_cache_mutex
            ),
            "singlecamera_tracker": FunctionNode(simulator, "singlecamera_tracker"),
            "associator": FunctionNode(simulator, "associator"),
            "geometric_properties": FunctionNode(simulator, "geometric_properties"),
            "geometry_filter": FunctionNode(simulator, "geometry_filter"),
            "multicamera_tracker": MutexFunctionNode(
                simulator, "multicamera_tracker", simulator.mt_mutex
            ),
            "tag_integrator": FunctionNode(simulator, "tag_integrator"),
            "rebroadcaster": FunctionNode(simulator, "rebroadcaster"),
            "detector_result_join": BufferNode(
                simulator,
                "detector_result_join",
                buffer_size=3 if simulator.new_module_enabled else 2,
            ),
            "gate_frame_selector": FrameSelectorNode(
                simulator, "gate_frame_selector", 1
            ),
            "gate_state_detector": FunctionNode(simulator, "gate_state_detector"),
            "gate_rebroadcaster": FunctionNode(simulator, "gate_rebroadcaster"),
        }

        self.nodes["decoder"].add_successor(self.nodes["frame_corrector"])
        self.nodes["frame_corrector"].add_successor(self.nodes["frame_to_detector"])
        self.nodes["frame_corrector"].add_successor(self.nodes["frame_selector"])
        self.nodes["frame_selector"].add_successor(0, self.nodes["nn_cache_searcher"])
        self.nodes["frame_selector"].add_successor(1, self.nodes["rebroadcaster"])
        self.nodes["nn_cache_searcher"].add_successor(
            0, self.nodes["nn_inputs_preparator"]
        )
        self.nodes["nn_cache_searcher"].add_successor(
            1, self.nodes["singlecamera_tracker"]
        )
        self.nodes["nn_inputs_preparator"].add_successor(self.nodes["nn_inferencer"])
        self.nodes["nn_inferencer"].add_successor(self.nodes["nn_postprocessor"])
        self.nodes["nn_postprocessor"].add_successor(self.nodes["nn_cache_writer"])
        self.nodes["nn_cache_writer"].add_successor(self.nodes["singlecamera_tracker"])
        self.nodes["singlecamera_tracker"].add_successor(self.nodes["associator"])
        self.nodes["associator"].add_successor(self.nodes["geometric_properties"])
        self.nodes["geometric_properties"].add_successor(self.nodes["geometry_filter"])
        self.nodes["geometry_filter"].add_successor(self.nodes["multicamera_tracker"])
        self.nodes["multicamera_tracker"].add_successor(self.nodes["tag_integrator"])
        self.nodes["tag_integrator"].add_successor(self.nodes["rebroadcaster"])
        self.nodes["rebroadcaster"].add_successor(self.nodes["detector_result_join"])
        self.nodes["frame_to_detector"].add_successor(
            self.nodes["detector_result_join"]
        )

        if simulator.new_module_enabled:
            self.nodes["frame_corrector"].add_successor(
                self.nodes["gate_frame_selector"]
            )
            self.nodes["gate_frame_selector"].add_successor(
                0, self.nodes["gate_state_detector"]
            )
            self.nodes["gate_frame_selector"].add_successor(
                1, self.nodes["gate_rebroadcaster"]
            )
            self.nodes["gate_state_detector"].add_successor(
                self.nodes["gate_rebroadcaster"]
            )
            self.nodes["gate_rebroadcaster"].add_successor(
                self.nodes["detector_result_join"]
            )

    def start(self):
        for node in self.nodes.values():
            node.process_tasks()


class VideoStream:
    def __init__(self, simulator: Simulator, total_frames: int) -> None:
        self.simulator = simulator
        self.total_frames = total_frames
        self.process_frames = 0
        self.flow_graph = VideoStreamFlowGraph(simulator)
        self.start_time = None
        self.finish_time = None

    def start(self):
        self.flow_graph.start()
        self.simulator.env.process(self._process())

    def _process(self):
        self.start_time = self.simulator.now()

        while self.process_frames != self.total_frames:
            yield self.simulator.env.process(self._process_single_frame())

        self.finish_time = self.simulator.now()

    def _process_single_frame(self):
        task = Store(self.simulator.env)
        # yield self.simulator.env.timeout(self.simulator.distributions["decoder"]())
        self.flow_graph.nodes["decoder"].task_queue.put(task)

        yield task.get()
        self.process_frames += 1

    def fps(self):
        return self.total_frames / (self.finish_time - self.start_time) * 1000
