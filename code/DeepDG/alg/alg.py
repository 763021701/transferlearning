# coding=utf-8
from alg.algs.ERM import ERM
from alg.algs.MMD import MMD
from alg.algs.MMD2 import MMD2
from alg.algs.CORAL import CORAL
from alg.algs.CORAL2 import CORAL2
from alg.algs.DANN import DANN
from alg.algs.RSC import RSC
from alg.algs.Mixup import Mixup
from alg.algs.MLDG import MLDG
from alg.algs.GroupDRO import GroupDRO
from alg.algs.ANDMask import ANDMask
from alg.algs.VREx import VREx
from alg.algs.DIFEX import DIFEX
from alg.algs.CFD import CFD
from alg.algs.CFD2 import CFD2

ALGORITHMS = [
    'ERM',
    'Mixup',
    'CORAL',
    'CORAL2',
    'MMD',
    'MMD2',
    'DANN',
    'MLDG',
    'GroupDRO',
    'RSC',
    'ANDMask',
    'VREx',
    'DIFEX',
    'CFD',
    'CFD2'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
