import networkx as nx
import numpy as np

#TODO: kirchhoff index per biggest graph per frame (at diff thresholds)


#TODO:return H for a given couple of i and j fragments at a given frame
def find_coupling(fragments, H):
    """Find Hij value between two fragments over the trajectory

    Parameters
    -------
    fragments : atomgroup

    Returns
    -------
    Hij_list: list of reals
      list runs over the trajectory
    average_Hij: real
      average over all timesteps
    """
    return coupling

#TODO: return number of nearest neighbors per fragment per frame
def nearneigh(fragments, detail=True):
    """Find nearest neighbors per fragment per frame

    Parameters
    -------
    fragments : atomgroup
    detail=True for full output (otherwise only average value is printed)

    Returns
    -------
    nn_list : dictionary of lists
      dictionary runs over fragments, list runs over trajectory
    average_nn: real
      average over all fragments and timesteps
    """

    return nn_list
#return number of contacts per fragment per frame
def contacts(fragments, thresh, graphs, full=True):
    """Find contacts per fragment per frame

    Parameters
    -------
    fragments : atomgroup

    full :  optional
      True =  full output
      False = only average value is returned

    Returns
    -------
    nc_list : dictionary of lists
      dictionary runs over fragments, list runs over trajectory
    average_nc: real
      average over all fragments and timesteps
    """

    return average_nc, n_contacts

def find_networks(fragments, H, degeneracy, thresh_list):
    """Find connectivity matrix

    Parameters
    ----------
    fragments : list of atomgroups
      contains coordinates of every molecule in the system
    H : np.array
      Hamiltonian matrix between fragment states
    degeneracy : int
      number of states for each fragment
    thresh_list : list of float
      different threshold values to build networks for

    Returns
    -------
    graphs : dictionary of list of nx.Graph
      for each threshold value, returns a list of networkx Graphs
      refer to fragment ids
    """

    # turns list of fragments into numpy array to allow indexing
    frag_arr = np.empty(len(fragments), dtype=object)
    frag_arr[:] = fragments

    # remove diagonal from H
    H_ij=H
    np.fill_diagonal(H_ij, 0)

    graphs = {}

    for thresh in thresh_list:
        g = nx.Graph()
        #TODO edge values should be abs(H)
        g.add_nodes_from(frag_arr)

        i, j = np.where(abs(H) > thresh)
        # divide indices by degeneracy to find fragment id
        i //= degeneracy
        j //= degeneracy

        frag_i = frag_arr[i]
        frag_j = frag_arr[j]

        # add fragments to a network if their H is above threshold
        g.add_weighted_edges_from(zip(frag_i, frag_j, np.abs(H_ij[i,j].real)),
                                  weight='Hij')

        # returns a dictionary where thresholds are keys, and values are
        # sorted lists of networks (the biggest is first)
        graphs[thresh] = sorted(nx.connected_component_subgraphs(g),
                              key = lambda x: len(x),
                              reverse = True)
    return graphs
