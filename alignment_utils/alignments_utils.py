import random
import re
from collections import deque
from math import sqrt

from pm4py import Marking
from pm4py.objects.log.obj import Trace
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

import petri_net_utils.petri_net_utils
from alignment_utils import a_star_modified
from petri_net_utils.gdt_spn import GDTSPN
from petri_net_utils.petri_net_utils import (enabled_transitions, convert_to_petri_net, fire_transition_on_copy,
                                             find_transition_by_label)
from scipy.stats import norm


def perform_alignment(trace: Trace, gdt_spn: GDTSPN, initial_marking, final_marking, DEBUG=False):
    if DEBUG:
        model, im, fm = convert_to_petri_net(gdt_spn, initial_marking, final_marking,
                                             eliminate_immediate_transitions=False)
    else:
        model, im, fm = convert_to_petri_net(gdt_spn, initial_marking, final_marking,
                                             eliminate_immediate_transitions=True)
    # visualizer.apply(model).view()
    # initial_marking = get_initial_marking(model)
    # final_marking = get_final_marking(model)
    model_cost_function = dict()
    trace_cost_function = list()
    sync_cost_function = dict()
    for t in model.transitions:
        # if the label is not None, we have a visible transition
        if t.label is not None and t.label != '':
            # associate cost 1 to each move-on-model associated to visible transitions
            model_cost_function[t] = 1  # ToDo: this should be zero for acyclic models
            # associate cost 0 to each sync move
            sync_cost_function[t] = 0
        else:
            # associate cost 1 to each move-on-model associated to hidden transitions
            model_cost_function[t] = 1
    for _ in range(len(trace)):
        trace_cost_function.append(1000)

    parameters = {}
    parameters[
        alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
    parameters[
        alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function
    parameters[
        alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function
    parameters[
        alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE] = True

    # alignment_candidates = alignment_utils.a_star_modified.apply(trace=trace, petri_net=model, initial_marking=im,
    #                             final_marking=fm, parameters=parameters)
    alignment, sync_prod, sync_initial_marking, sync_final_marking, cost_function = a_star_modified.apply(trace=trace,
                                                                                                          petri_net=model,
                                                                                                          initial_marking=im,
                                                                                                          final_marking=fm,
                                                                                                          parameters=parameters)

    ali_candidates = calc_all_candidates(sync_initial_marking, sync_final_marking, alignment['cost'], cost_function)

    return ali_candidates


def calc_all_candidates(initial_marking: Marking, final_marking: Marking, target_costs: int, cost_function: dict):
    """
     Breadth-First-Search algorithm to find all paths with target_costs to reach the final marking.
     starting from the initial marking
     :param initial_marking: The initial marking of a synchronous product
     :param final_marking: The final marking of a synchronous product
     :param target_costs: The minimal costs, that are required to fire a sequence of transitions,
     to get from the initial marking to the final marking.
     :param cost_function: A dict, that maps each transition of the synchronous product to the cost of firing it.
     :return: Returns a list of all possible cost minimal paths to fire a sequence of transitions from initial
     to final marking.
     """

    # Initialize the queue with (current_marking, current_cost, path)
    path_candidates = list()
    queue = deque([(initial_marking, 0, [])])
    visited = dict()

    while queue:
        current_marking, current_cost, path = queue.popleft()
        # check if target_costs are already reached
        # check if final marking is already reached
        if current_marking == final_marking and current_cost == target_costs:
            path_candidates.append(path)
            continue
        if current_cost > target_costs:
            continue
        # Explore all possible transitions
        enabled_transitions_set = enabled_transitions(current_marking)
        for transition in enabled_transitions_set:
            new_marking = fire_transition_on_copy(current_marking, transition)
            cost = cost_function[transition]
            new_cost = current_cost + cost
            # Only consider this new marking if it improves or matches the cost to reach it
            if new_marking not in visited or visited[new_marking] >= new_cost:
                visited[new_marking] = new_cost
                queue.append((new_marking, new_cost, path + [transition]))

    return path_candidates


def __find_transition_by_description(model, transition_description):
    transition_name = transition_description.name[1]
    # by convention (t_1, t_2) in sync_prod: t_1 is from trace_net, t_2 is from model
    for t in model.transitions:
        if t.name == transition_name:
            return t


def pick_alignment_old(gdt_spn: GDTSPN, alignment_candidates: list[list], alignment_as_transitions=True) -> tuple[
    list, int]:
    probabilities = set()
    list_index = 0
    for ali in alignment_candidates:
        probability = 1.0
        cleaned_alignment = list()
        for event in ali:
            curr_transition = __find_transition_by_description(gdt_spn, event)
            if alignment_as_transitions:
                cleaned_alignment.append(curr_transition)
            else:
                cleaned_alignment.append(curr_transition.label)
            probability = probability * curr_transition.weight
        probabilities.add((tuple(cleaned_alignment), probability,
                           list_index))  # list index to restore the original alignment later on
        list_index += 1
    # choose the best alignment
    # first criteria: greatest probability
    best = probabilities.pop()
    for candidate in probabilities:
        if candidate[1] > best[1]:
            best = candidate
    return list(best[0]), best[2]


def pick_alignment_new(gdt_spn: GDTSPN, alignment_candidates: list[list], trace, mode="geometric", threshold=0.05) -> tuple[list, int]:
    """
     Chooses the best alignment based on the time gaps and activities that the alignment proposes to reconstruct
     :param gdt_spn: The GDTSPN Model
     :param alignment_candidates: A list of cost-minimal alignments
     :param trace: A trace with timestamps
     :param mode: The method to calculate an overall score for an alignment
     :param threshold: Choose randomly the best candidates with a gap fitness at max worse than threshold.
                       Works like a tolerance
     :return: The chosen best-fitting alignment and the index of the chosen alignment
     """

    alignment_gap_fitness_dict = dict()
    for alignment_id in range(len(alignment_candidates)):
        clean_alignment = cleanup_alignment(alignment_candidates[alignment_id],
                                            cleanup_option='no_t_labels_in_alignments')
        gap_fitness = calc_full_gap_fitness(clean_alignment, trace, gdt_spn, mode)
        alignment_gap_fitness_dict[alignment_id] = gap_fitness
    best_alignment_id = max(alignment_gap_fitness_dict, key=lambda k: alignment_gap_fitness_dict[k])
    best_gap_fitness = alignment_gap_fitness_dict[best_alignment_id]

    best_candidates_in_threshold = list()
    for candidate_id in range(len(alignment_candidates)):
        if alignment_gap_fitness_dict[candidate_id] >= best_gap_fitness - threshold:
            best_candidates_in_threshold.append(alignment_candidates[candidate_id])
    chosen_candidate = random.choice(best_candidates_in_threshold)

    # transform alignment
    cleaned_alignment = transform_alignment_from_sync_to_gdt_spn(chosen_candidate, gdt_spn)
    return cleaned_alignment, best_alignment_id


def transform_alignment_from_sync_to_gdt_spn(alignment, gdt_spn):
    cleaned_alignment = list()
    for event in alignment:
        curr_transition = __find_transition_by_description(gdt_spn, event)
        cleaned_alignment.append(curr_transition)
    return cleaned_alignment


def calc_full_gap_fitness(alignment: list, trace: Trace, model: GDTSPN, mode):
    """
     Calculates the full gap fitness score for a given alignment
     :param alignment: The alignment to calculate the full gap fitness for
     :param trace: A trace with timestamps
     :param model: The GDTSPN Model
     :param mode: The method to calculate an overall score for an alignment
     :return: The overall gap fitness score for a given alignment
     """

    # determine how many gaps the candidate has
    gaps_list = find_all_gaps(alignment)
    # calc gap-fitness for each gap
    gap_fitness_values = list()
    for i in range(len(gaps_list)):
        gap_fitness = calc_gap_fitness_val(alignment, gaps_list[i][0], gaps_list[i][1], trace, model)
        gap_fitness_values.append(gap_fitness)
    full_gap_fitness = calculate_end_score(gap_fitness_values, mode)
    return full_gap_fitness


def calculate_end_score(gap_fitness_values, mode="geometric", epsilon=0.00001):
    """
     Calculates the end score for a list of given gap fitness value
     :param gap_fitness_values:
     :param mode: The method to calculate an overall score for an alignment
     :param epsilon: The epsilon parameter, if a single value is zero
     :return: The aggregated gap fitness score for a given alignment
     """

    if mode == "geometric":
        product = 1
        for val in gap_fitness_values:
            if epsilon > 0 and val == 0:
                product *= val + epsilon
            else:
                product *= val
        return product**(1/len(gap_fitness_values))
    elif mode == "arithmetic":
        my_sum = 0
        for val in gap_fitness_values:
            my_sum += val
        return my_sum / len(gap_fitness_values)
    elif mode == "harmonic":
        my_sum = 0
        for val in gap_fitness_values:
            if val == 0:
                my_sum += 1 / (val + epsilon)
            else:
                my_sum += 1 / val
        return len(gap_fitness_values) / my_sum
    elif mode == "multiply":
        product = 1
        for val in gap_fitness_values:
            if epsilon > 0 and val == 0:
                product *= val + epsilon
            else:
                product *= val
        return product
    else:
        raise NotImplemented("Only 4 modes are supported")


def find_all_gaps(alignment):
    """
     Identifies all gaps, i.e. all model moves, in a given alignment
     :param alignment: The alignment to find all gaps from
     :return: A list of tuples, consisting of the start index and the length of a gap
    """

    index = 0
    gaps_list = list()  # gaps are stored as tuples of (index_in_alignment, length)
    for ali in alignment:
        if ali is not alignment[index]:
            continue
        if ali.label[0] == ali.label[1]:
            index += 1
            continue
        else:
            # assume that no log moves are possible
            gap_start_index = index
            length = 0
            while index < len(alignment) and (alignment[index].label[0] == ">>" or alignment[index].label[1] == ">>"):
                length += 1
                index += 1
                if index >= len(alignment):
                    break
            gaps_list.append((gap_start_index, length))
            if index >= len(alignment):
                break
    return gaps_list


def calc_gap_fitness_val(alignment, gap_start_index, gap_length, trace, model):
    """
     Calculates the gap fitness for a specific gap within an alignment
     :param alignment: The alignment that contains the gap to calculate the gap fitness for
     :param gap_start_index: The start index of the gap
     :param gap_length: The length of the gap
     :param trace: A trace with timestamps
     :param model: The GDTSPN Model
     :return: The gap fitness score for a given gap within an alignment
    """

    # get gap_transitions from alignment and gap_index
    gap_transitions = get_gap_transitions(alignment, gap_start_index, gap_length, model)

    # get prev_gap transitions
    prev_gap_transitions, relative_ts_index = get_prev_gap_transitions(alignment, gap_start_index, model)
    if relative_ts_index == -1:
        relative_ts = 0  # case: first event in alignment is being reconstructed
    else:
        relative_ts = get_timestamp_by_alignment_step(alignment, trace, relative_ts_index)

    # get succ_transition
    successor_transition = None
    if len(alignment) > gap_start_index + gap_length:
        succ_label = alignment[gap_start_index + gap_length].label
        succ_transition = find_transition_by_label(model, succ_label[0])
        succ_transition_predecessors = petri_net_utils.petri_net_utils.get_predecessors(succ_transition, find_immediate_joins=False)
        if alignment[gap_start_index + gap_length - 1].label[0] == ">>":
            last_gap_t_label = alignment[gap_start_index + gap_length - 1].label[1]
        else:
            last_gap_t_label = alignment[gap_start_index + gap_length - 1].label[0]
        last_gap_transition = find_transition_by_label(model, last_gap_t_label)
        if last_gap_transition in succ_transition_predecessors:
            successor_transition = succ_transition

    # build normal distribution
    mean = 0
    var = 0
    for gap_t in gap_transitions:
        mean += gap_t.time_performance['mean']
        var += gap_t.time_performance['variance']
    if prev_gap_transitions is not None:
        for t in prev_gap_transitions:
            mean += t.time_performance['mean']
            var += t.time_performance['variance']
    if successor_transition is not None:
        mean += successor_transition.time_performance['mean']
        var += successor_transition.time_performance['variance']

    # temporal boundaries
    first_gap_t = gap_transitions[0]
    left = 0
    if gap_start_index > 0:
        alignment_predecessor_label = alignment[gap_start_index - 1].label[0]
        alignment_predecessor = find_transition_by_label(model, alignment_predecessor_label)
        gap_causal_dependencies = petri_net_utils.petri_net_utils.get_predecessors(first_gap_t, find_immediate_joins=False)

        if alignment_predecessor not in gap_causal_dependencies:
            # case: if AP is concurrent to first gap transition
            ap_ts = get_timestamp_by_alignment_step(alignment, trace, gap_start_index-1)
            difference = ap_ts - relative_ts
            left = difference.total_seconds()

    if len(alignment) > gap_start_index + gap_length:
        succ_ts = get_timestamp_by_alignment_step(alignment, trace, gap_start_index+gap_length)
        difference = succ_ts - relative_ts
        right = difference.total_seconds()
    else:
        difference = trace[-1]['time:timestamp'] - relative_ts  # some (relatively big number)
        right = difference.total_seconds()

    # integrate for cumulative probability
    values = [left, right]
    cumulative_probabilities = norm.cdf(values, mean, sqrt(var))
    area = abs(cumulative_probabilities[0] - cumulative_probabilities[1])
    gap_fitness_value = calc_gap_fitness_from_integral(area)
    return gap_fitness_value


def calc_gap_fitness_from_integral(cumulative_probability):
    """
     Calculates the gap fitness score of a given cumulative probability for a gap
     :param cumulative_probability: the cumulative probability
     :return: the gap_fitness_value
    """
    return 1 - abs(2 * cumulative_probability - 1)


def get_gap_transitions(alignment, gap_start_index, gap_length, model):
    """
     Collects all the gap transitions for a given gap as transition objects
     :param alignment: The alignment that contains the gap to calculate the gap fitness for
     :param gap_start_index: The start index of the gap
     :param gap_length: The length of the gap
     :param model: The GDTSPN Model
     :return: The gap transitions
    """

    transition_labels = list()
    for i in range(gap_length):
        t_label = alignment[gap_start_index + i].label
        assert ">>" in t_label[0] or ">>" in t_label[1]  # gap must start with model move
        transition_labels.append(t_label)
    transition_list = list()
    for transition_label in transition_labels:
        if transition_label[0] == ">>":
            gap_transition = find_transition_by_label(model, transition_label[1])
        else:
            gap_transition = find_transition_by_label(model, transition_label[0])
        transition_list.append(gap_transition)
    return transition_list


def get_prev_gap_transitions(alignment, gap_start_index, model):
    """
     Collects all transitions that are needed before the actual gap
     :param alignment: The alignment
     :param gap_start_index: The start index of the gap
     :param model: The GDTSPN Model
     :return: A list of all transitions that are needed before the gap
    """

    # find all predecessors P that are timed transitions
    # look at the alignment predecessor AP. if AP is in P, then causal transition is found.
    # if AP has no timestamp i.e. is also a model move, then repeat for the same for this transition
    if gap_start_index == 0:
        return None, -1
    transition_list = list()
    if ">>" in alignment[gap_start_index].label[0]:
        first_gap_label = alignment[gap_start_index].label[1]
    else:
        first_gap_label = alignment[gap_start_index].label[0]
    first_gap_t = find_transition_by_label(model, first_gap_label)
    index = gap_start_index - 1
    curr_transition_label = alignment[index].label
    curr_transition = find_transition_by_label(model, curr_transition_label[0])  # direct predecessor is always sync move
    timed_predecessors = petri_net_utils.petri_net_utils.get_predecessors(first_gap_t, find_immediate_joins=False)
    while curr_transition not in timed_predecessors:
        if index <= 0:
            return transition_list, -1
        index -= 1
        curr_transition_label = alignment[index].label
        if curr_transition_label[0] == ">>":
            curr_transition = find_transition_by_label(model, curr_transition_label[1])
        else:
            curr_transition = find_transition_by_label(model, curr_transition_label[0])
        if curr_transition in timed_predecessors and (alignment[index].label[0] == ">>" or alignment[index].label[1] == ">>"):  # only add model moves
            transition_list.append(curr_transition)
    else:
        curr_t_is_model_move = curr_transition_label[0] == ">>" or curr_transition_label[1] == ">>"
        if curr_t_is_model_move:
            # found causal dependency but it is a model move
            # now go backwards until sync move is found for a timestamp (needed later)
            # add all model moves until then
            while curr_t_is_model_move:
                if index <= 0:
                    return transition_list, -1
                index -= 1
                curr_transition_label = alignment[index].label
                if curr_transition_label[0] == ">>":
                    curr_transition = find_transition_by_label(model, curr_transition_label[1])
                else:
                    curr_transition = find_transition_by_label(model, curr_transition_label[0])
                if alignment[index].label[0] == ">>" or alignment[index].label[1] == ">>":
                    transition_list.append(curr_transition)
                else:
                    curr_t_is_model_move = False

    relative_ts_index = index
    return transition_list, relative_ts_index


def get_timestamp_by_alignment_step(alignment, trace, index):
    """
     Retrieves the timestamp of a given alignment step
     :param alignment: The alignment
     :param trace: The trace
     :param index: The index of the activity that the timestamp belongs to
     :return: The timestamp of the activity that the timestamp belongs to
    """

    label = alignment[index].label[0]
    count = 0
    for i in range(index + 1):
        if i == 0:
            continue
        if alignment[index - i].label[0] == label:  # found another ali_step with same label
            if alignment[index - i].label[0] == alignment[index - i].label[1]:
                # ignore model moves, they do not have timestamps
                count += 1
    counter2 = 0
    for event in trace:
        event_activity = event['concept:name']
        if event_activity == label:
            if counter2 != count:
                counter2 += 1
            else:
                return event['time:timestamp']


def cleanup_alignment(alignment: list, cleanup_option='no_immediate_transitions'):
    """
     Eliminates all Immediate Transitions from an alignment
     :param alignment: The alignment
     :param cleanup_option: The cleanup option. Either to eliminate ImmediateTransitions or
            synced transitions that contain t_labels
     :return: The alignment without Immediate Transitions
    """
    if cleanup_option == 'no_immediate_transitions':
        cleaned_alignment = list()
        for event in alignment:
            if not isinstance(event, GDTSPN.ImmediateTransition):
                cleaned_alignment.append(event)
        return cleaned_alignment
    elif cleanup_option == 'no_t_labels_in_alignments':
        cleaned_alignment = list()
        for event in alignment:
            pattern = r'^t\d+(_\d+)?'  # t followed by a number. possibly followed by "_" and a number again
            if ">>" in event.label[0]:
                if not re.match(pattern, event.label[1]):
                    cleaned_alignment.append(event)
            else:
                if not re.match(pattern, event.label[0]):
                    cleaned_alignment.append(event)
        return cleaned_alignment
    return None


def flatten_alignment(alignment):
    """
     Flattens the alignment to just consists of labels.
     :param alignment: The alignment
     :return: The flattened alignment consisting only of labels.
    """
    flattened_alignment = list()
    for event in alignment:
        if isinstance(event.label, tuple):
            if event.label[0] == event.label[1]:
                flattened_alignment.append(event.label[0])
            elif event.label[0] == ">>":
                flattened_alignment.append(event.label[1])
            elif event.label[1] == ">>":
                flattened_alignment.append(event.label[0])
        else:
            flattened_alignment.append(event.label)
    return flattened_alignment


def check_model_moves(alignment):
    """
     Checks, if alignment contains at least one model move to qualify for repair
     :param alignment: The alignment
     :return: True, if the alignment contains at least one model move
    """
    for event in alignment:
        if isinstance(event.label, tuple):
            if event.label[0] == ">>" or event.label[1] == ">>":
                return True
    return False



