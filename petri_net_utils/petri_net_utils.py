from collections import deque
from copy import copy

import pm4py.objects
from pm4py import Marking, PetriNet
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to, add_transition, add_place

from petri_net_utils.gdt_spn import GDTSPN


def find_transition_by_label(model, label: str):
    for t in model.transitions:
        if t.label == label:
            return t


def get_places_from_marking(curr_marking):
    places = set()
    for elem in curr_marking.keys():
        if curr_marking[elem] > 0:
            places.add(elem)
    return places


def check_path_for_timed_transitions(path):
    """
     Checks, whether any Transition on the path is a TimedTransition. Used for Unfolding.
     :param path: A list of transitions that represents a path through a net
     :return False, if any Transition on the path is a TimedTransition. True, otherwise.
     """
    path.pop()
    for elem in path:
        if isinstance(elem, GDTSPN.TimedTransition):
            return False
    return True


def find_path_to_next_transition(wanted_transition: GDTSPN.Transition, start_place):
    """
     Finds a path from start place to wanted_transition using Breadth-First-Search.
     :param start_place: The starting place node.
     :param wanted_transition: The target transition node.
     :return: A list of transitions forming the path if found, otherwise None.
     """

    # Initialize the queue with the start place
    queue = deque([(start_place, [])])
    visited = set()

    while queue:
        current_node, path = queue.popleft()

        # Check if the current node is a place or a transition
        if isinstance(current_node, GDTSPN.Place):
            # Move to each transition connected by an outgoing arc
            for arc in current_node.out_arcs:
                if arc.target not in visited:
                    visited.add(arc.target)
                    queue.append((arc.target, path))
        elif isinstance(current_node, GDTSPN.Transition):
            # Append the transition to the path
            new_path = path + [current_node]

            # If the target transition is found, return the path
            if current_node == wanted_transition:
                # check path has no TimedTransitions
                if check_path_for_timed_transitions(new_path.copy()):
                    return new_path
                else:
                    continue

            # Otherwise, move to each place connected by an outgoing arc
            for arc in current_node.out_arcs:
                if arc.target not in visited:
                    visited.add(arc.target)
                    queue.append((arc.target, new_path))

    return None  # No path found


def add_elements_to_net(net, path, last_net_element, occurrences):
    """
     Adds the transitions in path one by one starting from last_net_element
     :param net: The net the elements are added to.
     :param path: A list of transitions to add.
     :param last_net_element: The place in the net at which the path is being added to.
     :param occurrences: keeps track of how often a transition/place was being used during path (for correct labelling)
     """
    split_count = 0  # count how many places should be added in case of a split
    prev_elem = None
    for elem in path:
        if occurrences[elem.label] > 1:
            new_name = elem.name + "_" + str(occurrences[elem.label])
            new_label = elem.label + "_" + str(occurrences[elem.label])
        else:
            new_name = elem.name
            new_label = elem.label
        if isinstance(elem, GDTSPN.ImmediateTransition):
            new_elem = GDTSPN.ImmediateTransition(new_name, new_label, None, None, elem.weight)
            net.transitions.add(new_elem)
        elif isinstance(elem, GDTSPN.TimedTransition):
            new_elem = GDTSPN.TimedTransition(new_name, new_label, None, None,
                                              elem.weight, elem.time_performance)
            net.transitions.add(new_elem)
        assert new_elem
        if prev_elem:
            # add as many places in between as the previous transition puts token in to
            for _ in range(split_count):
                new_place = GDTSPN.Place("p_" + str(len(net.places)))
                net.places.add(new_place)
                add_arc_from_to(prev_elem, new_place, net)
            else:
                add_arc_from_to(new_place, new_elem, net)
        else:
            # place to transition
            # should be all places without postset, if net has no nesting
            if len(elem.in_arcs) == 1:
                add_arc_from_to(last_net_element, new_elem, net)
            else:
                places_to_connect = set()
                for p in net.places:
                    if len(p.out_arcs) == 0:
                        places_to_connect.add(p)
                for p in places_to_connect:
                    add_arc_from_to(p, new_elem, net)

        prev_elem = new_elem
        split_count = len(elem.out_arcs)

    # add a place after last transition
    new_place = GDTSPN.Place("p_" + str(len(net.places)))
    net.places.add(new_place)
    add_arc_from_to(prev_elem, new_place, net)
    ret_val = set()
    ret_val.add(new_place)
    return ret_val


def enabled_transitions(marking: Marking):
    """
     Creates a set of transitions that are enabled at a specific marking
     :param marking: The current marking
     :return Set of transitions, that are enabled within the marking
     """
    enabled_transitions_set = set()
    for p in marking.keys():
        out_arcs = p.out_arcs
        for a in out_arcs:
            if transition_is_activated(marking, a.target):
                enabled_transitions_set.add(a.target)
    return enabled_transitions_set


def transition_is_activated(marking, transition):
    """
     Checks, whether a transition is activated at a marking
     :param marking: The current marking
     :param transition: The transition that is being checked
     :return True, if the transition is activated. False, otherwise.
     """
    in_arcs_set = transition.in_arcs
    pre_set = [arc.source for arc in in_arcs_set]
    for p in pre_set:
        if p not in marking.keys() or marking[p] <= 0:
            return False
    return True


def fire_transition(marking, transition):
    """
     Fires a transition given a marking
     :param marking: The current marking
     :param transition: The transition that is being fired
     """
    in_arcs_set = transition.in_arcs
    pre_set = [arc.source for arc in in_arcs_set]
    for p in pre_set:
        marking[p] -= 1
        if marking[p] == 0:
            marking.__delitem__(p)
    out_arcs_set = transition.out_arcs
    post_set = [arc.target for arc in out_arcs_set]
    for p in post_set:
        if p not in marking.keys():
            marking[p] = 1
        else:
            marking[p] += 1


def fire_transition_on_copy(marking, transition):
    """
     Fires a transition given a marking. Operates on a copy of the given marking, to not alter the original marking.
     :param marking: The current marking
     :param transition: The transition that is being fired
     :return A copy of the given marking after the transition was fired.
     """
    new_marking = copy(marking)
    fire_transition(new_marking, transition)
    return new_marking


def replay_path(marking, firing_sequence):
    """
     Fires a sequence of transitions given a marking
     :param marking: The current marking
     :param firing_sequence: A list of transitions that will be fired step-by-step
     """
    for transition in firing_sequence:
        if transition_is_activated(marking, transition):
            fire_transition(marking, transition)
        else:
            raise Exception("firing sequence is not possible to fire!")


def update_occurrences(firing_sequence, label_occurrences):
    """
     Updates for a given sequence of transitions, how often a label was observed during firing
     :param firing_sequence: A list of transitions
     :param label_occurrences: A dict, that keeps track of how often labels were seen, when transitions were fired
     """
    for transition in firing_sequence:
        if transition.label in label_occurrences.keys():
            label_occurrences[transition.label] += 1
        else:
            label_occurrences[transition.label] = 1


def adapt_path_for_net(net: GDTSPN, path: list[GDTSPN.Transition]):
    """
     Given a list of transitions from another net, this functions constructs an adapted list of transitions that are
     associated to the provided net. The matching works by comparing the the labels.
     :param net: The net, where the adapted path is constructed.
     :param path: A list of transitions that are associated with another net.
     :return A new path (list of transitions), that contains the same transitions but in the given net.
     """
    fixed_path = list()
    for path_transition in path:
        candidates = set()
        for t in net.transitions:
            if t.label.startswith(path_transition.label):
                candidates.add(t)
        candidates_count = len(candidates)
        if candidates_count == 1:
            fixed_path.append(candidates.pop())
        elif candidates_count == 2:
            best_candidate = candidates.pop()
            label_parts = best_candidate.label.split("_")
            if len(label_parts) == 1:   # this is the original transition, e.g. 't1' and has no suffix
                best_candidate = candidates.pop()  # so the other one must be wanted transition
            fixed_path.append(best_candidate)
        else:
            # remove original transition, e.g. 't1' with no suffix
            for c in candidates:
                if len(c.label.split("_")) == 1:
                    break
            candidates.remove(c)
            best_candidate = candidates.pop()
            candidates_count -= 2  # original and this one
            label_parts = best_candidate.label.split("_")
            bc_suffix = label_parts[-1]
            for _ in range(candidates_count):
                next = candidates.pop()
                next_label_suffix = (next.label.split("_"))[-1]
                if int(next_label_suffix) > int(bc_suffix):
                    best_candidate = next
            fixed_path.append(best_candidate)
    return fixed_path


def get_equivalent_place_from_markings(marking: Marking, orig_place: GDTSPN.Place):
    """
     Given a place from another marking, this functinon finds the euqivalent place in the provided marking by comparing
     the place name.
     :param marking: The marking, that contains a similar place
     :param orig_place: The place from another marking, that has to be found in the given marking
     :return A similar place as orig_place but from the provided marking.
     """
    unfolded_net_place_set = get_places_from_marking(marking)
    orig_label_set = set()
    for in_arc in orig_place.in_arcs:
        orig_label_set.add(in_arc.source.label.split("_")[0])

    for place in unfolded_net_place_set:
        new_place_label_set = set()
        for in_arc in place.in_arcs:
            new_place_label_set.add(in_arc.source.label.split("_")[0])
        if new_place_label_set.issubset(orig_label_set) and len(place.out_arcs) == 0:
            return place


def get_initial_marking(net):
    """
     Provides the initial marking of a net, given that each place without a preset is a starting place.
     :param net: A petri net
     :return An initial marking of the net based on the assumption above.
     """
    initial_marking = Marking()
    initial_place = None
    for p in net.places:
        if len(p.in_arcs) == 0:
            initial_place = p
    assert initial_place is not None
    initial_marking[initial_place] = 1
    return initial_marking


def get_final_marking(net: PetriNet):
    """
     Provides the final marking of a net, given that each place without a postset is an ending place.
     :param net: A petri net
     :return A final marking of the net based on the assumption above.
     """
    final_marking = Marking()
    final_place = None
    for p in net.places:
        if len(p.out_arcs) == 0:
            final_place = p
    assert final_place is not None
    final_marking[final_place] = 1
    return final_marking


def convert_to_petri_net(gdt_spn_model: GDTSPN, initial_marking, final_marking, eliminate_immediate_transitions=False):
    """
     Converts a GDT_SPN model into a standard Petri Net. Therefore, each Timed and each Immediate Transition will be
     converted to a regular Petri Net Transition.
     :param gdt_spn_model: The GDT_SPN net that will be converted
     :param initial_marking: The initial marking of the given net
     :param final_marking: The final marking of the given net
     :param eliminate_immediate_transitions: A (debug) flag, that preserves immediate transitions, if activated.
     :return A standard petri net with the same language as the given net.
     """
    petri_net = PetriNet(gdt_spn_model.name + "_conv")
    obj_map = dict()  # maps transitions/places from the original net to the new net transitions/places for correct arcs
    if eliminate_immediate_transitions:
        for t in gdt_spn_model.transitions:
            if not isinstance(t, GDTSPN.ImmediateTransition):
                obj_map[t] = add_transition(petri_net, t.name, t.label)
        for p in gdt_spn_model.places:
            obj_map[p] = add_place(petri_net, p.name)
        for a in gdt_spn_model.arcs:
            if isinstance(a.target, GDTSPN.ImmediateTransition):
                if len(a.target.out_arcs) == 1:  # TODO parallel split! TODO parallel join!
                    out = next(iter(a.target.out_arcs))
                    next_place = out.target
                    out2 = next(iter(next_place.out_arcs))
                    next_transition = out2.target  # get the only next transition
                    add_arc_from_to(obj_map[a.source], obj_map[next_transition], petri_net)
                    # remove unused places
                    pm4py.objects.petri_net.utils.petri_utils.remove_place(petri_net, a.target.out_arcs.pop())
    else:
        for t in gdt_spn_model.transitions:
            obj_map[t] = add_transition(petri_net, t.name, t.label)
        for p in gdt_spn_model.places:
            obj_map[p] = add_place(petri_net, p.name)
        for a in gdt_spn_model.arcs:
            add_arc_from_to(obj_map[a.source], obj_map[a.target], petri_net)

    # copy the markings
    im = Marking()
    for p in initial_marking:
        im[obj_map[p]] = initial_marking[p]
    fm = Marking()
    for p in final_marking:
        fm[obj_map[p]] = final_marking[p]
    return petri_net, im, fm


def get_successors(t: GDTSPN.Transition):
    successors = set()
    if len(t.out_arcs) == 0:
        raise Exception("Net must not end with an transition!")
    elif len(t.out_arcs) == 1:
        next_place = next(iter(t.out_arcs)).target
        if len(next_place.out_arcs) == 0:
            return None
        elif len(next_place.out_arcs) == 1:
            next_transition = next(iter(next_place.out_arcs)).target
            if isinstance(next_transition, GDTSPN.TimedTransition):
                successors.add(next_transition)
                return successors
            elif isinstance(next_transition, GDTSPN.ImmediateTransition):
                return get_successors(next_transition)
        else:
            raise Exception("This Net should be choice-free. So places do not have more than one outgoing arc")
    elif len(t.out_arcs) == 2:  # split of two
        if isinstance(t, GDTSPN.ImmediateTransition):
            next_places = set()
            for arc in t.out_arcs:
                next_places.add(arc.target)
            next_transitions = set()
            for p in next_places:
                next_transitions.add(next(iter(p.out_arcs)).target)  # works because each place has a maximum of one out_arc
            for next_t in next_transitions:
                if isinstance(next_t, GDTSPN.TimedTransition):
                    successors.add(next_t)
                elif isinstance(next_t, GDTSPN.ImmediateTransition):
                    successors.union(get_successors(next_t))
                else:
                    raise Exception("Net must consist only of timed and immediate transitions!")
        else:
            raise Exception("Only Immediate Transitions may have multiple outgoing arcs")
    else:
        raise NotImplemented("More than a split of two paths is not implemented yet")
    return successors


def get_predecessors(t: GDTSPN.Transition, find_immediate_joins=True):
    predecessors = set()
    if len(t.in_arcs) == 0:
        raise Exception("Net must not end with an transition!")
    elif len(t.in_arcs) == 1:
        prev_place = next(iter(t.in_arcs)).source
        if len(prev_place.in_arcs) == 0:
            return None
        elif len(prev_place.in_arcs) == 1:
            prev_transition = next(iter(prev_place.in_arcs)).source
            if isinstance(prev_transition, GDTSPN.TimedTransition):
                predecessors.add(prev_transition)
                return predecessors
            elif isinstance(prev_transition, GDTSPN.ImmediateTransition):
                if len(prev_transition.in_arcs) == 2 and find_immediate_joins:
                    # ImmediateTransition is needed for BN if it is joining two incoming branches
                    predecessors.add(prev_transition)
                    return predecessors
                else:
                    return get_predecessors(prev_transition, find_immediate_joins)
        else:
            raise Exception("Unfolded nets have 1 incoming arc at max")
    elif len(t.in_arcs) == 2:  # join of two paths
        if isinstance(t, GDTSPN.ImmediateTransition):
            prev_places = set()
            for arc in t.in_arcs:
                prev_places.add(arc.source)
            prev_transitions = set()
            for p in prev_places:
                if find_immediate_joins:  # mode for unfolding
                    prev_transitions.add(next(iter(p.in_arcs)).source)
                else:  # mode for gap fitness
                    for a in p.in_arcs:
                        prev_transitions.add(a.source)
            for prev_t in prev_transitions:
                if isinstance(prev_t, GDTSPN.TimedTransition):
                    predecessors.add(prev_t)
                elif isinstance(prev_t, GDTSPN.ImmediateTransition):
                    if len(prev_t.in_arcs) == 2 and find_immediate_joins:
                        predecessors.add(prev_t)
                    else:
                        predecessors.union(get_predecessors(prev_t, find_immediate_joins))
                else:
                    raise Exception("Net must consist only of timed and immediate transitions!")
        else:
            raise Exception("Only Immediate Transitions may have multiple incoming arcs")
    else:
        raise NotImplemented("More than a join of two paths is not implemented yet")
    return predecessors


def unfold_spn_with_alignment(gdt_spn: GDTSPN, initial_marking: Marking, alignment: list[str]):
    """
     Unfolds the GDT_SPN according to an alignment. The resulting net will be acyclic and can fully replay the alignment.

     :param gdt_spn: The GDT_SPN will be unfolded
     :param initial_marking: The initial marking of the GDT_SPN net
     :param alignment: A list of transition labels according to which the unfolding is being performed
     :return: The unfolded GDT_SPN
     """
    unfolded_net = GDTSPN(gdt_spn.name + "_unfolded")
    orig_net_marking = initial_marking
    unfolded_net_marking = Marking()
    label_occurrences = {}
    last_element = None
    for i in range(len(alignment)):
        next_transition = find_transition_by_label(gdt_spn, alignment[i])
        curr_places = get_places_from_marking(orig_net_marking)
        for p in curr_places:
            # search a path from a curr_place to next transition according to alignment
            path = find_path_to_next_transition(next_transition, p)
            if path:
                break
        assert path and p
        update_occurrences(path, label_occurrences)
        # add this path to the unfolded model
        last_element = get_equivalent_place_from_markings(unfolded_net_marking, p)  # Todo: make this a set of elements?
        if i == 0:  # only for i == 0
            for place in curr_places:
                new_place = GDTSPN.Place(place.name)
                unfolded_net.places.add(new_place)
                unfolded_net_marking[new_place] = 1
            add_elements_to_net(unfolded_net, path, new_place, label_occurrences)

        else:
            add_elements_to_net(unfolded_net, path, last_element, label_occurrences)
        # visualizer.apply(unfolded_net).view()
        # simulate the firing sequence (keep track of current markings in original and unfolded net)
        replay_path(orig_net_marking, path)
        path_for_new_net = adapt_path_for_net(unfolded_net, path)
        replay_path(unfolded_net_marking, path_for_new_net)
    return unfolded_net
