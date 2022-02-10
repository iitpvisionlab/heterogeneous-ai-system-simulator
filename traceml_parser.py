import xml.etree.ElementTree as ET
from collections import defaultdict


def _parse_node_names_single_file(filename: str) -> dict:
    node_names = {}

    tree = ET.parse(filename)
    root = tree.getroot()
    for node in root.findall("./graph/node"):
        data = node.find("./data[@key='d0']")

        node_id = node.attrib["id"]
        node_name = data.text
        node_names[node_id] = node_name

    return node_names


def parse_node_names(graphml_filenames: list) -> dict:
    node_names = {}

    for filename in graphml_filenames:
        node_names_single = _parse_node_names_single_file(filename)
        node_names = {**node_names, **node_names_single}

    return node_names


def _find_thread_count(root: ET.Element) -> int:
    for attribute in root.findall("./description/attribute"):
        if attribute.attrib["name"] == "Thread":
            return len(list(attribute))

    raise ValueError("No Thread attribute found")


def parse_traceml(filename: str, node_names: dict) -> dict:
    task_durations = defaultdict(list)

    tree = ET.parse(filename)
    root = tree.getroot()

    thread_count = _find_thread_count(root)

    thread_current_task = [[]] * thread_count

    for node in root:
        if not "tid" in node.attrib:
            continue

        thread_id = int(node.attrib["tid"])

        if node.tag == "task_begin":
            task_id = node.attrib["id"]
            start_timestamp = int(node.attrib["ts"])
            thread_current_task[thread_id].append((task_id, start_timestamp))
        elif node.tag == "task_end":
            assert len(thread_current_task[thread_id]) > 0

            end_timestamp = int(node.attrib["ts"])
            task_id, start_timestamp = thread_current_task[thread_id].pop()

            duration = end_timestamp - start_timestamp
            assert duration >= 0

            task_durations[task_id].append(duration)

    node_durations = defaultdict(list)
    for task_id, durations in task_durations.items():
        node_name = node_names[task_id] if task_id in node_names else task_id
        node_durations[node_name] += durations

    return dict(node_durations)
