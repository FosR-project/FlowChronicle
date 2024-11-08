import math
import scipy

def length_natural(x:int,c = 2.865064) -> float:
     assert x > 0 # only defined for positive integers
     bits = math.log2(c)
     xf = float(x)
     while xf>1:
         xf = math.log2(xf)
         bits += xf
     return bits


def log_choose(n:int, k:int) -> float:
    return math.log2(scipy.special.binom(n,k))