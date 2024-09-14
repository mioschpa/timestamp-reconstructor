import re
from copy import copy
from datetime import timedelta

from pm4py import Marking
from pm4py.objects.log.obj import Trace, Event
from alignment_utils.alignments_utils import perform_alignment, pick_alignment_new, cleanup_alignment, \
    flatten_alignment, check_model_moves, transform_alignment_from_sync_to_gdt_spn
from bayesian_utils.bayesian_utils import perform_inference, start_engine, terminate_engine, \
    add_bayes_net_toolbox, prepare_bnt, construct_bayesian_network_new, prepare_evidence, add_evidence_for_max_nodes
from petri_net_utils.gdt_spn import GDTSPN
from petri_net_utils.petri_net_utils import unfold_spn_with_alignment


def reconstruct_timestamp_with_spn(trace: Trace, gdt_spn: GDTSPN, gdt_spn_im: Marking, gdt_spn_fm: Marking):
    # 1. Repair structure
    # 1.1 perform alignment
    alignment_candidates = perform_alignment(trace, gdt_spn, copy(gdt_spn_im), gdt_spn_fm, DEBUG=True)
    # 1.2 pick alignment
    if len(alignment_candidates) == 1:
        alignment = transform_alignment_from_sync_to_gdt_spn(alignment_candidates[0], gdt_spn)
        candidate_id = 0
        if False and not check_model_moves(alignment_candidates[0]):  # TODO buggy behavior
            # if alignment does not contain any model moves, then no repair is needed, return same trace
            return trace
    else:
        alignment, candidate_id = pick_alignment_new(gdt_spn, alignment_candidates, trace, mode="geometric")
    alignment = cleanup_alignment(alignment, 'no_immediate_transitions')
    alignment = flatten_alignment(alignment)

    # 2. Repair time
    # 2.1 unfold spn according to alignment
    unfolded_net = unfold_spn_with_alignment(gdt_spn, copy(gdt_spn_im), alignment)
    # 2.2 construct bayesian network
    bn, predecessors = construct_bayesian_network_new(unfolded_net, copy(gdt_spn_im))
    engine = start_engine()
    add_bayes_net_toolbox(engine)
    node_dict = prepare_bnt(engine, bn, alignment, predecessors)
    evidence = prepare_evidence(trace, alignment, node_dict, alignment_candidates[candidate_id])
    evidence = add_evidence_for_max_nodes(evidence, bn, node_dict)
    missing_nodes = get_missing_node_ids(node_dict, evidence)  # needed to query missing timestamps
    # 2.3 insert evidence into bayesian network and perform inference
    timestamps, variance_list = perform_inference(engine, evidence, missing_nodes)
    # 2.4 add missing times to added entries
    event_timestamps = transform_into_event_timestamps(timestamps, node_dict)
    # terminate_engine(engine)
    reconstructed_trace = insert_missing_timestamps(trace, alignment_candidates[candidate_id], event_timestamps)
    return reconstructed_trace


def transform_into_event_timestamps(timestamps, node_dict):
    """"
     Creates a human-readable dictionary that maps node labels to timestamps
    :param timestamps: a dictionary that maps a node id to a timestamp
    :param node_dict: a dictionary that maps a node label to an id
    :return: a dictionary that maps node labels to timestamps
    """
    event_timestamps = dict()
    for node in node_dict.keys():
        node_id = node_dict[node]
        if node_id not in timestamps:
            continue
        event_timestamps[node] = timestamps[node_id]
    return event_timestamps


def get_missing_node_ids(node_dict: dict, evidence: dict):
    node_ids = set(node_dict.values())
    evidence_node_ids = set(evidence.keys())
    missing_node_ids = node_ids.difference(evidence_node_ids)
    return missing_node_ids


def insert_missing_timestamps(trace, original_alignment, timestamps):
    reconstructed_trace = Trace()

    # denormalize timestamp again
    initial_timestamp = trace[0]["time:timestamp"]

    # filter out the join nodes (e.g. t5, or t5_2, etc.)
    filtered_timestamps = {key: value for key, value in timestamps.items() if not re.match(r'^t\d+', key)}
    sorted_timestamps = sorted(round(value, 2) for value in filtered_timestamps.values())

    # extract known timestamps
    known_timestamps = list()
    for event in trace:
        event_ts = event['time:timestamp']
        known_timestamps.append(event_ts)

    ts_index = 0  # iterate over sorted timestamps
    trace_index = 0  # iterate over the original trace

    for transition in original_alignment:
        composed_labels = transition.label
        # if one of both labels is >> and the other indicates an immediate Transition,
        # then iterate until TimedTransition
        if ">>" in composed_labels and any(re.match(r"t\d+$", label) for label in composed_labels):
            continue

        if composed_labels[0] == composed_labels[1]:  # this is a known event with known timestamp
            reconstructed_trace.append(trace[trace_index])
            trace_index += 1

        else:  # this is an added/reconstructed event whose timestamp has to be injected
            event_name = composed_labels[0]
            if event_name == '>>':
                event_name = composed_labels[1]
            reconstructed_trace.append(Event({"concept:name": event_name,
                                              "time:timestamp": initial_timestamp
                                              + timedelta(seconds=sorted_timestamps[ts_index])}))
            ts_index += 1
    for event in reconstructed_trace:
        print(event)

    return reconstructed_trace
