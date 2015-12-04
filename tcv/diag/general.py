# CLASS containing the set of fromtree to access some general
# time traces like current, bt, elongation, triangularity, line-average density
# central density from thomson, central temperature from thomson, central temperature from
# double filter

class General(object):
    """
    Python class to access some general signals for TCV. Implemented method (as @classmethod) are
    ip : read plasma current
    bt : read the magnetic field on axis
    neline: line average density
    ne0: central density from Thomson
    te0: Central temperature from Thomson
    betapol: poloidal beta
    q: q. Can load q0 or q95 depending on call
    delta: Triangularity. 
    kappa:
    """
