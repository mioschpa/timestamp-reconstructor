import re

import numpy as np
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork
from scipy.stats import norm
from pm4py import Marking

import matlab.engine

from petri_net_utils.gdt_spn import GDTSPN
from petri_net_utils.petri_net_utils import get_predecessors


def start_engine():
    return matlab.engine.start_matlab()


def terminate_engine(engine):
    return engine.quit()


def add_bayes_net_toolbox(eng):
    bnt_path = r'C:\\Users\\Admin\\Downloads\\bnt-master\\'
    eng.cd(bnt_path)
    eng.addpath(bnt_path, nargout=0)

    bnt_full_path = eng.genpathKPM(bnt_path, nargout=1)
    eng.addpath(bnt_full_path, nargout=0)


def approximate_two_gaussians_mult(X1_mu, X1_sigma, X2_mu, X2_sigma, corr_coeff=0):
    """
    Calculates the first two moments of the maximum of two normal distributions.
    Returns an approximate normal distribution with these moments.

    Parameters:
    :param X1_mu: Mean of the first normal distribution
    :param X1_sigma: Standard deviation of the first normal distribution
    :param X2_mu: Mean of the second normal distribution
    :param X2_sigma: Standard deviation of the second normal distribution
    :param corr_coeff: Correlation coefficient between the two distributions (default is 0)

    :return max_mean: Approximate mean of the maximum
    :return max_variance: Approximate standard deviation of the maximum
    """
    theta = np.sqrt(X1_sigma**2 + X2_sigma**2 - 2 * corr_coeff * X1_sigma * X2_sigma)
    z = (X1_mu - X2_mu) / theta 

    # first moment is mean
    max_mean = X1_mu * norm.cdf(z) + X2_mu * (1 - norm.cdf(z)) + theta * norm.pdf(z)

    # second moment is variance
    max_variance = (
            (X1_sigma ** 2 + X1_mu ** 2) * norm.cdf(z) +
            (X1_sigma ** 2 + X2_mu ** 2) * (1 - norm.cdf(z)) +
            (X1_mu + X2_mu) * theta * norm.pdf(z)
    )
    return max_mean, np.sqrt(max_variance)


def construct_bayesian_network_new(spn: GDTSPN, initial_marking: Marking):
    """
     Constructs a bayesian network using an unfolded GDT_SPN net.

     :param spn: An unfolded GDT_SPN net. The net must be acyclic and choice-free.
     :param initial_marking: A marking object representing the initial marking
     :return: A bayesian network that represents the GDT_SPN net.
     """
    # start_place = get_start_place(initial_marking)
    bn = LinearGaussianBayesianNetwork()

    # Idee
    # create a graph consisting of transitions
    needed_transitions = [t for t in spn.transitions if isinstance(t, GDTSPN.TimedTransition) or len(t.in_arcs) == 2]
    predecessors = dict()
    # successors = dict()
    for t in needed_transitions:
        predecessors[t] = get_predecessors(t)
        # successors[t] = get_successors_new(t)
    # create with this information a BNGraph with its relations
    for t in needed_transitions:
        bn.add_node(t.label)
    for t in predecessors.keys():
        predecessor_set = predecessors[t]
        if predecessor_set is None:
            continue
        for pred in predecessor_set:
            bn.add_edge(pred.label, t.label)
    for t in predecessors.keys():
        predecessor_set = predecessors[t]
        if predecessor_set is None:
            cpd = LinearGaussianCPD(t.label, [t.time_performance['mean']], t.time_performance['variance'])
            bn.add_cpds(cpd)
        elif len(predecessor_set) == 1:  # one dependency
            dependency = next(iter(predecessor_set))
            cpd = LinearGaussianCPD(t.label, [t.time_performance['mean'], 1],
                                    t.time_performance['variance'], [dependency.label])
            bn.add_cpds(cpd)
        else:  # multiple dependencies at joins
            if len(predecessor_set) == 2:
                dependencies = list(predecessor_set)
                norm_dist1 = dependencies[0].time_performance
                norm_dist2 = dependencies[1].time_performance
                approx_max = approximate_two_gaussians_mult(norm_dist1['mean'], norm_dist1['variance'],
                                                            norm_dist2['mean'], norm_dist2['variance'])
                dep_labels = [dep.label for dep in predecessor_set]
                cpd = LinearGaussianCPD(t.label, [approx_max[0], 0, 0], approx_max[1], dep_labels)
                bn.add_cpds(cpd)
            else:
                raise NotImplementedError("A BN Node with 3 dependencies is not implemented yet!")
    return bn, predecessors


def map_node_to_id(alignment, predecessors):
    """
     Creates a dictionary that creates a dictionary that maps each node to an ID in the correct order according to the
     alignment.

     :param alignment: The chosen alignment that prescribes the correct ordering of the IDs
     :param predecessors: A dictionary that knows the predecessors of each node in the unfolded GDT_SPN net. Used for
     adding joining t_X nodes that are not part of the alignment
     :return: A dictionary that maps each node to an ID in same order as they are in the bayesian network.
     """
    node_dict = dict()
    joining_nodes = dict()
    for key in predecessors.keys():
        if predecessors[key] and len(predecessors[key]) == 2:
            joining_nodes[key] = predecessors[key]

    i = 1  # index starts at 1 because of MATLAB notation
    for event in alignment:
        add_join_node_flag = False
        event_label = event
        if event_label in node_dict:
            # create new label with (biggest) index
            suffix = 1
            old_label = event_label
            while True:
                suffix += 1
                event_label = f"{old_label}_{suffix}"
                if event_label not in node_dict.keys():
                    break
        assert event_label not in node_dict.keys()
        # if curr_label is predecessor of t_X node:
        for key in joining_nodes.keys():
            labels = [t.label for t in joining_nodes[key]]  # get the two predecessor labels
            if labels[0] in node_dict and labels[1] == event_label:  # both predecessors have to be added beforehand
                # add t_x afterward
                add_join_node_flag = True
                break
            elif labels[1] in node_dict and labels[0] == event_label:  # both predecessors have to be added beforehand
                # add t_x afterward
                add_join_node_flag = True
                break
        node_dict[event_label] = i
        i += 1
        if add_join_node_flag:
            node_dict[key.label] = i
            i += 1
    return node_dict


def prepare_bnt(engine, bayesian_network: LinearGaussianBayesianNetwork, alignment, predecessors):
    # create dag
    n = bayesian_network.number_of_nodes()
    node_dict = map_node_to_id(alignment, predecessors)
    # node_dict = dict()
    # for index, node in enumerate(bayesian_network.nodes):
       # node_dict[node] = index+1
    dag = engine.zeros(n, n)
    engine.workspace['dag'] = dag
    # engine.desktop(nargout=0)
    # manipulate dag
    for edge in bayesian_network.edges:
        engine.eval("dag(" + str(node_dict[edge[0]]) + "," +  str(node_dict[edge[1]]) + ") = 1;", nargout=0)
    ns = engine.ones(1, n)
    engine.workspace['ns'] = ns
    dag = engine.workspace['dag']
    bnet = engine.mk_bnet(dag, ns, 'discrete', [])
    engine.workspace['bnet'] = bnet
    for cpd in bayesian_network.cpds:
        node_id = node_dict[cpd.variable]
        command = "bnet.CPD{" + str(node_id) + "} = gaussian_CPD(bnet," + str(node_id) + ", 'mean', " + str(
            cpd.mean[0]) + ", 'cov', " + str(cpd.variance) + ");"
        if len(cpd.evidence) == 1:   # if node has one parent
            command = command[:-2] + ", 'weights', 1);"
        elif len(cpd.evidence) == 2:  # node has two parents
            command = command[:-2] + ", 'weights', [1, 1]);"

        engine.eval(command, nargout=0)
    return node_dict


def prepare_evidence(trace, alignment, node_dict, original_alignment):
    """
    Extracts the known timestamps from a trace to be used as evidence for a bayesian network
    :param trace: A trace with known timestamps
    :param alignment:
    :param node_dict: A dictionary to get for each activity the label of the BN node
    : param original_alignment: The original alignment of the synchronous product
    :return: Returns a dictionary that contains the timestamp for each event
    """
    evidence = dict()
    times = list()
    for event in trace:
        times.append(event['time:timestamp'])

    normalized_times = [0]  # normalize times, such that first event completes at t = 0

    for i in range(1, len(times)):
        time_diff = (times[i] - times[i - 1]).total_seconds()
        new_timestamp = normalized_times[-1] + time_diff
        normalized_times.append(new_timestamp)

    activity_counter = dict()

    # TODO: what if missing activity is the first? shift all events then!
    i = 0  # iterate over the known times
    orig_pos = 0
    for activity in alignment:
        actual_activity = activity
        composed_labels = original_alignment[orig_pos].label
        # if one of both labels is >> and the other indicates an immediate Transition, then skip till TimedTransition
        while ">>" in composed_labels and any(re.match(r"t\d+$", label) for label in composed_labels):
            orig_pos += 1
            composed_labels = original_alignment[orig_pos].label

        if activity in activity_counter:
            activity_counter[activity] += 1
            actual_activity = f"{activity}_{activity_counter[activity]}"
        else:
            activity_counter[activity] = 1
        if composed_labels[0] == composed_labels[1]:
            if activity == composed_labels[0]:  # This is evidence!
                node_id = node_dict[actual_activity]
                if node_id not in evidence.keys():
                    evidence[node_id] = normalized_times[i]
                else:  # code unreachable? possible bug here
                    # choose next activity: e.g. A_2
                    if re.match(r".*?_\d+$", str(activity)):
                        # Split the string on the last underscore
                        base, num = activity.rsplit('_', 1)
                        # Increment the number
                        incremented_num = int(num) + 1
                        # Return the base of the string with the incremented number
                        new_activity = base + '_' + str(incremented_num)
                    else:
                        new_activity = activity + "_2"
                    node_id = node_dict[new_activity]
                    evidence[node_id] = normalized_times[i]
                i += 1  # used a known timestamp, increment to get the next timestamp later
                orig_pos += 1  # sync move
        else:
            # model move - i.e. a missing event in the original trace
            orig_pos += 1
    return evidence


def add_evidence_for_max_nodes(evidence: dict, bayesian_network: LinearGaussianBayesianNetwork, node_dict: dict):
    """
     Calculates all max nodes if evidence exists for both parents of a max node

     :param evidence: the evidence dictionary that contains the timestamps from the given trace
     :param bayesian_network: the bayesian network
     :param node_dict: the dictionary that maps each transition label to a node id in the bayesian network
    """

    # find all max nodes
    for node in bayesian_network.nodes:
        parents = bayesian_network.get_parents(node)
        if len(parents) == 0 or len(parents) == 1:
            continue
        # get parents of max node
        parents_id = [node_dict[parent] for parent in parents]
        timestamps = list()
        # check if evidence exists for both parents
        for parent in parents_id:
            if not evidence.__contains__(parent):
                break
            timestamps.append(evidence[parent])
        else:
            # calculate the max of both
            max_of_evidence = max(timestamps)
            # set the evidence for a max node
            evidence[node_dict[node]] = max_of_evidence
    return evidence


def perform_inference(engine, evidence: dict, query_nodes: set):
    number_of_nodes = engine.workspace['ns'].size[1]
    engine.eval("engine = jtree_inf_engine(bnet);", nargout=0)
    engine.eval("evidence = cell(1," + str(number_of_nodes) + ");", nargout=0)
    # insert evidence
    for key in evidence.keys():
        value = evidence[key]
        engine.eval("evidence{" + str(key) + "} = " + str(value) + ";", nargout=0)
    engine.eval("[engine, loglik] = enter_evidence(engine, evidence);", nargout=0)
    marginals_list = dict()
    variance_list = dict()
    for query in query_nodes:
        engine.eval("marginals = marginal_nodes(engine," + str(query) + ");", nargout=0)
        marginals = engine.workspace['marginals']
        variance_list[query] = marginals['Sigma']
        marginals_list[query] = marginals['mu']
    return marginals_list, variance_list


