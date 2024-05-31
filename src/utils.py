import os
import re

LIB_FILE = './lib/7nm/7nm.lib'

def obtain_aig(state: str):
    """
    Obtain the current AIG from state.
    Provided by the project proposal.

    Parameters:
    state(str): e.g. "alu2_0130622"
    """
    
    circuitName, actions = state.split('_')
    circuitPath = './dataset/InitialAIG/train/' + circuitName + '.aig'
    
    logFile = circuitName + '.log'
    nextState = state + '.aig'
    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }
    actionCmd = ''
    for action in actions:
        actionCmd += (synthesisOpToPosDic[int(action)] + '; ')
    abcRunCmd = "./yosys -abc -c \"read " + circuitPath + "; " + actionCmd + "read_lib " + LIB_FILE + "; write " + nextState + "; print_stats\" > " + logFile

    os.system(abcRunCmd)
    return logFile, nextState

def eval_aig(AIG: str):
    """
    Evaluate the AIG with Yosys
    Provided by the project proposal.
    """

    logFile = AIG.split(".")[0] + ".log"
    abcRunCmd = "./yosys -abc -c \"read " + AIG + "; read_lib " + LIB_FILE + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    eval = float(areaInformation[-9]) * float(areaInformation[-4])
