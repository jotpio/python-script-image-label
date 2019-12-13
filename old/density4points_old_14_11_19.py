import numpy as np
import h5py

def LoadFilterChains(FILE, LENGTH_THRESHOLD):
    '''
    OUTPUT:
        chains.shape(N, time, pos)
    '''
    data = h5py.File(FILE)
    heads = np.array(data['chainPositionJoined'])
    tails = np.array(data['chainOrientationVectorJoined'])
    data.close()
    keeper = np.zeros(len(heads[:, 0]), dtype=int)
    i = 0
    for head in heads:
        if len(np.where(head[:, 0] > 0)[0]) >= LENGTH_THRESHOLD:
            keeper[i] = 1
        i += 1
    heads = heads[keeper == 1]
    tails = tails[keeper == 1]
    return heads, tails


def LoadChainsInFrame(FILE, FRAME, LENGTH_THRESHOLD):
    '''
    - load chains which exists in frame "FRAME".
    - at time points where they do not exist -> set to nan 
    OUTPUT:
        chains.shape(Time, N, 2)
    '''
    chains, tails = LoadFilterChains(FILE, FRAME, LENGTH_THRESHOLD)
    # select only chains which are present at t=FRAME
    there = np.where(chains[:, FRAME, 0] > 0)[0]
    chains = chains[there]
    tails = tails[there]
    # set non-existing positions to NaN
    there0 = np.where(chains[:, :, 0] <= 0)
    chains[there0] = np.nan
    tails[there0] = np.nan
    return chains, tails


def LoadPositions(FILE, FRAME, LENGTH_THRESHOLD):
    '''
    OUTPUT:
        heads.shape(Time, N, 2)
    '''
    if type(FRAME) == int:
        FRAME = [FRAME]
    Pos = np.ones(len(FRAME), dtype='object')
    heads, _ = LoadFilterChains(FILE, LENGTH_THRESHOLD)
    posi = np.transpose(heads, (1, 0, 2))  # change dimension (N, Time, Pos)->(Time, N, Pos)
    # only take positions of fish which are present in current frame:
    for i in range(len(FRAME)):
        pos = posi[FRAME[i]]
        Pos[i] = pos[pos[:, 0] > 0]
    return Pos


def LoadPositionsHeadings(FILE, FRAME, LENGTH_THRESHOLD):
    '''
    OUTPUT:
        heads.shape(Time, N, 2)
    '''
    if type(FRAME) == int:
        FRAME = [FRAME]
    Pos = np.ones(len(FRAME), dtype='object')
    Ori = np.ones(len(FRAME), dtype='object')
    heads, tail = LoadFilterChains(FILE, LENGTH_THRESHOLD)
    # change dimension (N, Time, Pos)->(Time, N, Pos)
    posi = np.transpose(heads, (1, 0, 2))
    ori = np.transpose(-tail, (1, 0, 2))
    # only take positions of fish which are present in current frame:
    for i in range(len(FRAME)):
        pos = posi[FRAME[i]]
        orii = ori[FRAME[i]]
        Pos[i] = pos[pos[:, 0] > 0]
        Ori[i] = orii[pos[:, 0] > 0]
    return Pos, Ori


def get_nematic_tensor(phis, weights=None):
    '''
    Returns the nematic tensor for 2 dimensions -> Matrix
    INPUT:
        phis.shape(N)
            orientation of all N particles
    '''
    N = len(phis)
    if weights is None:
        weights = np.ones(N, dtype='float') / N
    assert np.round(weights.sum(), 5) == 1., 'weights must add up to 1, but its:' + str(weights.sum())
    Q = np.empty((2, 2), dtype='float')
    Q[0, 0] = np.dot(np.cos(2 * phis), weights)
    Q[1, 1] = np.dot((-1) * np.cos(2 * phis), weights)
    Q[0, 1] = np.dot(np.sin(2 * phis), weights)
    Q[1, 0] = Q[0, 1] * 1
    return Q


def get_nematics(phis, weights=False):
    '''
    Returns the nematic order and direction
    INPUT:
        phis.shape(N)
            orientation of all N particles
    '''
    N = len(phis)
    Q = get_nematic_tensor(phis, weights=weights)
    # np.linalg.eigh instead of np.linalg.eig (1:its symmetric matric, 2:wrong eigenvector for eigenvalue in linalg.eig)
    evals, evecs = np.linalg.eigh(Q)
    idx = np.argmax(evals)
    evec = evecs[idx]
    return [evals[idx], evecs[idx]]


def gaussian4points(points, pos, ori, std, limits=None):
    """
    computes density and other averages for specific points
    INPUT:
        points, shape=(N, 2) 
            x,y-positions of points where average gonna be computed
        pos.shape(N, 2)
            position of N individuals in this frame
        ori.shape(N, 2)
            orientation-vector of N individuals in this frame
        std double
            standard-deviation of gaussian-kernel
        limits [[minx, miny], [maxx, maxy]]
            defines limits of image(important to normalize density)
    OUTPUT:
        list [density, orientation, polarisation, n_orientation, n_polarisation]
            orientation.shape(N), ....    
    """
    if limits is None:
        limits = [pos.min(axis=0), pos.max(axis=0)]
    limits = np.array(limits)
    # create points of grid (to normalize)
    sizes = limits[1] - limits[0]
    ratio = sizes[0]/sizes[1]
    fine = 200
    gridx, gridy = np.meshgrid(np.linspace(limits[0, 0], limits[1, 0], int(ratio*fine)),
                               np.linspace(limits[0, 0], limits[1, 0], fine))
    grid = np.concatenate((gridx[:, :, None], gridy[:, :, None]),
                          axis=2).reshape(-1, 2)
    N, _ = points.shape
    # Create the empty holders.
    density = np.zeros(N, dtype=float)
    orientation = np.zeros(N, dtype=float)
    polarisation = np.zeros(N, dtype=float)
    n_polarisation = np.zeros(N, dtype=float)
    n_orientation = np.zeros(N, dtype=float)
    for i, point in enumerate(points):
        displacement = pos - point[None, :] 
        distance = np.sqrt(displacement[:, 0] ** 2 +
                           displacement[:, 1] ** 2)
        # Create gaussian-weight vector, with std=rad:
        weight = np.exp(-distance ** 2 / (2 * std ** 2))
        # density:
        # create an area equivalent quantitiy with same std
        grid_displacement = grid - point[None, :]
        grid_distance = np.sqrt(grid_displacement[:, 0] ** 2 +
                                grid_displacement[:, 1] ** 2)
        area_weight = np.exp(-grid_distance ** 2 / (2 * std ** 2))
        density[i] = weight.sum()/area_weight.sum()
        weight /= weight.sum() # normalize the weight
        # Polarisation and Average direction of motion (orientation)
        ori_v = ori * weight[:, None]
        group_vector = np.sum(ori_v, axis=0)
        assert len(group_vector) == 2
        orientation[i] = np.arctan2(group_vector[1], group_vector[0])
        polarisation[i] = np.sqrt(np.dot(group_vector, group_vector))

        # Nematic Order(use arctan instead of arctan2):
        ori1 = np.arctan2(ori[:, 1], ori[:, 0])
        nem_polarisation, nem_orientation = get_nematics(ori1, weights=weight)
        n_orientation[i] = np.arctan(nem_orientation[1] / nem_orientation[0])
        n_polarisation[i] = nem_polarisation
    return [density, orientation, polarisation, n_orientation, n_polarisation]
