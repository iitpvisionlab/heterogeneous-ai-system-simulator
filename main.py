#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import numpy as np

from distributions import make_empirical_distributions_from_traces
from simulator import NNCacheType, Simulator
from traceml_parser import parse_node_names, parse_traceml
from video_stream import VideoStream


def _parse_owlstopwatch_traces(path: Path):
    with open(path, "r") as file:
        traces = [int(line) * 1000 for line in file]

    return traces


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-cores", type=int, default=12)
    parser.add_argument("--video-streams", type=int, default=6)
    parser.add_argument("--total-frames", type=int, default=1464)
    parser.add_argument("--graphs", nargs="+", type=Path, required=True)
    parser.add_argument("--graph-traces", type=Path, required=True)
    parser.add_argument("--nn-inferencer-traces", type=Path, required=True)
    parser.add_argument("--decoder-traces", type=Path, required=True)
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--enable-new-module", action="store_true")
    parser.add_argument("--use-async", action="store_true")
    parser.add_argument(
        "--nn-cache",
        type=lambda x: NNCacheType[x],
        choices=list(NNCacheType),
        default=NNCacheType.NOT_ACTIVATED,
    )
    args = parser.parse_args()

    traces = parse_traceml(args.graph_traces, parse_node_names(args.graphs))
    nn_inferencer_tracers = _parse_owlstopwatch_traces(args.nn_inferencer_traces)
    decoder_traces = _parse_owlstopwatch_traces(args.decoder_traces)
    traces["decoder"] = decoder_traces
    traces["nn_compilation"] = nn_inferencer_tracers[
        0:1
    ]  # The first nn_inferencer trace is the compilation time
    traces["nn_inferencer"] = nn_inferencer_tracers[1:]
    # traces["gate_state_detector"] = [0.1e9]
    if args.nn_cache is NNCacheType.IDEAL_FULL:
        traces["nn_cache_searcher"] = [0]
        traces["nn_cache_writer"] = [0]

    distributions = make_empirical_distributions_from_traces(traces, args.uniform)

    results = []

    for seed in range(args.runs):
        random.seed(seed)
        simulator = Simulator(
            cpu_core_count=args.cpu_cores,
            distributions=distributions,
            nn_cache_type=args.nn_cache,
            new_module_enabled=args.enable_new_module,
            use_async = args.use_async
        )
        streams = [
            VideoStream(simulator, total_frames=args.total_frames)
            for _ in range(args.video_streams)
        ]
        for stream in streams:
            stream.start()
        simulator.run()

        streams_fps = [stream.fps() for stream in streams]
        average_fps = np.mean(streams_fps)
        results.append(average_fps)

    print(f"Average FPS: {np.mean(results):.3f}")


if __name__ == "__main__":
    _main()
