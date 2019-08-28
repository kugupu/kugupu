#    kugupu - molecular networks for change transport
#    Copyright (C) 2019  Micaela Matta and Richard J Gowers
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Network/Graph based operations

All 'matrices' are square dense numpy arrays

All 'graphs' are networkx.Graph objects

"""
import networkx as nx
import numpy as np


def adjacency_matrix(network, weighted=True):
    """Adjacency matrix from graph

    Parameters
    ----------
    network : nx.Graph
      graph representation of charge transport network
    weighted : bool, optional
      if True edges are weighted according to Hij,
      if False edges are all given a weight of 1.0

    Returns
    -------
    adj : numpy array
      adjacency matrix
    """
    if weighted:
        return nx.to_numpy_array(network, weight='Hij')
    else:
        return nx.to_numpy_array(network)


def laplacian_matrix(network, weighted=True):
    """Laplacian matrix

    Weights are taken from 'Hij' property of edges

    Parameters
    ----------
    network : nx.Graph

    Returns
    -------
    laplacian : numpy array
    """
    adj = adjacency_matrix(network, weighted=weighted)
    deg = adj.sum(axis=0)
    adj *= -1
    np.fill_diagonal(adj, deg)

    return adj


#TODO: kirchhoff index per biggest graph per frame (at diff thresholds)
def resistance_distance_matrix(network, weighted=True):
    """ Return resistance distance matrix

      RD[i,j] = S[i,i] + S[j,j] - 2S[i,j]

    Where S is the psuedo inverse of the Laplacian

    Parameters
    ----------
    network : nx.Graph

    Returns
    -------
    res_dist : numpy array
    """
    Q = np.linalg.pinv(laplacian_matrix(network, weighted=weighted))

    res_dist = np.zeros_like(Q)
    res_dist += Q.diagonal()
    res_dist += Q.diagonal().T[:, None]
    res_dist -= 2 * Q

    return res_dist


def admittance_distance_matrix(network, weighted=True):
    """Admittance distance matrix

       A[i,j]= 1/RD[i,j] if i!=j;
       0 if i=j

    as defined in eq. 4 of J. Phys. Chem. Lett. 2015, 6, 1018-21.
    """
    RD = resistance_distance_matrix(network, weighted=weighted)

    A = np.zeros_like(RD)
    A[np.nonzero(RD)] = 1 / RD[np.nonzero(RD)]

    return A


def kirchhoff_transport_index(network, weighted=True):
    """Kirchhoff 'transport index':

       Kt = sum A[i,j]/2N^2

    Where
     A : 2d numpy array, real
      admittance matrix as in eq. 4 of J. Phys. Chem. Lett. 2015, 6, 1018-21
     N : int
      network order

    To avoid double counting of A[i,j], since the matrix is symmetric,
    we normalize by 2N^2 (and not N^2 as in eq. 5 of
    J. Phys. Chem. Lett. 2015, 6, 1018-21).
    """
    A = admittance_distance_matrix(network, weighted=weighted)
    Kt = A.sum()
    Kt /= 2 * network.order()**2

    return Kt


def kirchhoff_index(network, weighted=True):
    """ Kirchhoff (Resistance) Index (Kf)

    Kf = 1/2 * sum_i sum_j RD[i,j]
    Where RD = resistance distance matrix

    """
    RD = resistance_distance_matrix(network, weighted=weighted)
    Kf = RD.sum() * 0.5

    return Kf


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


def H_address_to_fragment(degeneracy):
    """Create an array for mapping H_frag indices to fragment indices

    Parameters
    ----------
    degeneracy : numpy array
      number of degenerate states per fragment

    Returns
    -------
    mapping : numpy array
      array relating H_frag indices to fragment indices
      ie mapping[21] reveals which fragment index molecular
      orbital 21 belongs to.  mapping will have the length of
      the size of the H_frag matrix.
    """
    return np.repeat(np.arange(degeneracy.shape[0]), degeneracy)


def find_networks(fragments, H, degeneracy, threshold, e_tol=0.3):
    """Find molecular networks between fragments for given threshold

    Fragments i & j will be considered "connected" if:
      abs(H[i, j]) > threshold
    AND
      abs(H[i, i] - H[j, j]) < e_tol

    Parameters
    ----------
    fragments : list of AtomGroup
      will be used as node labels
    H : np.array
      Hamiltonian matrix between fragment states for a single
      time frame
    degeneracy : numpy array
      number of degenerate states for each molecule in fragments
    threshold : float
      threshold value to build networks for
    e_tol : float
      maximum allowed different in energy levels (in eV)
      between two states

    Returns
    -------
    graphs : list of nx.Graph
      returns a list of networkx Graphs, sorted according to descending
      graph size.  Each graph has AtomGroups as nodes, edges represent
      molecular coupling and are weighted in eV
    """
    # turns list of fragments into numpy array to allow fancy indexing
    frag_arr = np.empty(len(fragments), dtype=object)
    frag_arr[:] = fragments

    # we want copy of data as we modify diagonal
    H_ij = H.copy()
    # energy of each state
    energies = H.diagonal().copy()
    # remove diagonal from H
    np.fill_diagonal(H_ij, 0)

    g = nx.Graph()
    # all fragments are nodes
    g.add_nodes_from(frag_arr)
    # Now to find the edges...
    # indices of where H_ij is above threshold
    i, j = np.where(np.abs(H_ij) > threshold)
    # check that states are within a certain energy tolerance
    e_crit = np.abs(energies[i] - energies[j]) < e_tol
    # filter i-j pairs according to energy tolerance
    i, j = i[e_crit], j[e_crit]

    # convert MO index to fragment index
    # ie if molecules had multiple degeneracy,
    # this is where that gets squashed down into molecular basis
    H_map = H_address_to_fragment(degeneracy)
    i, j = H_map[i], H_map[j]
    frag_i, frag_j = frag_arr[i], frag_arr[j]

    # add fragments to a network if their H is above threshold
    # TODO: Combining multiple degeneracy edges?
    g.add_weighted_edges_from(zip(frag_i, frag_j, H_ij[i,j]),
                              weight='Hij')

    # returns a dictionary where thresholds are keys, and values are
    # sorted lists of networks (the biggest is first)
    graphs = sorted(nx.connected_component_subgraphs(g),
                    key=lambda x: len(x),
                    reverse=True)

    return graphs
