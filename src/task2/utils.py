import os
import re
import abc_py
import numpy as np
import torch

LIB_FILE = './lib/7nm/7nm.lib'
INIT_AIG_DIR = './InitialAIG/train/'


def obtain_aig(state: str):
    """
    Obtain the current AIG from state.\\
    Provided by the project proposal.

    Params:
    state(`str`): e.g. "alu2_0130622"
    """

    circuit_name, actions = state.split('_')
    circuit_path = INIT_AIG_DIR + circuit_name + '.aig'
    log_path = circuit_name + '.log'
    aig_path = state + '.aig'
    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }
    action_cmd = ''
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + '; ')
    abc_cmd = "yosys-abc -c \"read " + circuit_path + "; " + action_cmd + "read_lib " + LIB_FILE + "; write " + aig_path + "; print_stats\" > " + log_path

    os.system(abc_cmd)
    return log_path, aig_path


def aig_baseline(aig_path: str) -> float:
    circuit_name = aig_path.split('_')[0]
    circuit_path = INIT_AIG_DIR + circuit_name + '.aig'
    log_path = circuit_name + '.log'

    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    abcRunCmd = "yosys-abc -c \"read " + circuit_path + "; " + RESYN2_CMD + " read_lib " + LIB_FILE + "; write " + aig_path + "; map; topo; stime\" >" + log_path
    os.system(abcRunCmd)
    with open(log_path) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])
    return baseline


def eval_aig(aig_path: str) -> float:
    """
    Evaluate the AIG with Yosys.\\
    Provided by the project proposal.
    """

    circuit_name = aig_path.split('_')[0]
    circuit_path = INIT_AIG_DIR + circuit_name + '.aig'
    log_path = circuit_name + '.log'
    abc_cmd = "yosys-abc -c \"read " + aig_path + "; read_lib " + LIB_FILE + "; map; topo; stime\" > " + log_path
    os.system(abc_cmd)
    with open(log_path) as f:
        area_information = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    eval = float(area_information[-9]) * float(area_information[-4])

    baseline = aig_baseline(aig_path)
    return 1 - eval / baseline


def eval_decision(state: str) -> float:
    log_path, aig_path = obtain_aig(state)
    abcRunCmd = "yosys-abc -c \"read " + aig_path + "; read_lib " + LIB_FILE + "; map; topo; stime\" > " + log_path
    os.system(abcRunCmd)
    with open(log_path) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        adpVal = float(areaInformation[-9]) * float(areaInformation[-4])

    baseline = aig_baseline(aig_path)
    os.remove(log_path)
    os.remove(aig_path)
    return (baseline - adpVal) / baseline


def convert_aig_to_tensor(aig_path: str) -> dict:
    """
    Using abc_py to represent AIG through the node connectivity and the features for each code.\\
    Provided by the project proposal.

    Params:
    aig_path(`str`): path to AIG file

    Return:
    data(dict): pack of feature
    * num_nodes(`int`): number of nodes `N`
    * num_edges(`int`): number of edges `E`
    * node_type(`torch.Tensor`): type of each node, of size `N`
    * num_inverted_predecessors(`torch.Tensor`): \\
        each node's number of inverted predecessors, of size `N`
    * edge_index(`torch.Tensor`): edges, of size `2 * E`\\
        structured as [src_idxs, target_idxs]
    """

    _abc = abc_py.AbcInterface()
    _abc.start()
    _abc.read(aig_path)
    data = {}

    num_nodes = _abc.numNodes()
    data['node_type'] = np.zeros(num_nodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(num_nodes, dtype=int)
    edge_src_index = []
    edge_target_index = []

    for nodeIdx in range(num_nodes):
        aig_node = _abc.aigNode(nodeIdx)
        node_type = aig_node.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if node_type == 0 or node_type == 2:
            data['node_type'][nodeIdx] = 0
        elif node_type == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2
        if node_type == 4:
            data['num_inverted_predecessors'][nodeIdx] = 1
        if node_type == 5:
            data['num_inverted_predecessors'][nodeIdx] = 2
        if aig_node.hasFanin0():
            fanin = aig_node.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        if aig_node.hasFanin1():
            fanin = aig_node.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)

    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['num_nodes'] = num_nodes
    data['num_edges'] = len(edge_src_index)

    return data


def eval_aig_greedy(aig_path: str) -> float:
    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }
    log_path = aig_path.split(".")[0] + ".log"
    AIG = aig_path.split(".")[0] + "_"
    for step in range(10):
        childs = []
        for child in range(7):
            childFile = AIG + str(child) + ".aig"
            abcRunCmd = "yosys-abc -c \"read " + aig_path + "; " + synthesisOpToPosDic[
                child] + "; read_lib " + LIB_FILE + "; write " + childFile + "; print_stats\" > " + log_path
            os.system(abcRunCmd)
            childs.append(childFile)
        childScores = Evaluation(childs)
        action = argmax(childScores)
        AIG = AIG + str(action)
    AIG = AIG + ".aig"
    abcRunCmd = "yosys-abc -c \"read " + AIG + "; read_lib " + LIB_FILE + "; map; topo; stime\" > " + log_path
    os.system(abcRunCmd)
    with open(log_path) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        adpVal = float(areaInformation[-9]) * float(areaInformation[-4])

    circuit_path = INIT_AIG_DIR + aig_path.split(".")[0] + '.aig'
    logFile = aig_path.split(".")[0] + ".log"
    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    abcRunCmd = "yosys-abc -c \"read " + circuit_path + "; " + RESYN2_CMD + " read_lib " + LIB_FILE + "; write " + aig_path + "; map; topo; stime\" >" + logFile
    os.system(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])

    return (baseline - adpVal) / baseline