"""Controls running and parsing yaehmop tight binding calculations"""
from collections import Counter, namedtuple
from scipy import sparse
import numpy as np
import subprocess
import os
from tqdm import tqdm
from MDAnalysis.lib import distances

from . import logger


FragSizes = namedtuple('FragSizes', ['starts', 'stops', 'sizes', 'n_electrons'])


def create_bind_inp(dimer_ags, dimer_name):
    """Create YAEHMOP input file for a single dimer

    Parameters
    ----------
    dimer_ags : tuple of mda.AtomGroup
      the fragments to be ran
    dimer_name : tuple
      indices of the dimer

    Returns
    -------
    output : str
      contents of Yaehmop input
      can either be piped in or written to file
    """
    frag_i, frag_j = dimer_ags

    logger.debug("Checking if fragments are in correct image")
    c_i = frag_i.center_of_mass()
    c_j = frag_j.center_of_mass()

    tol = 0.1
    d1 = distances.calc_bonds(c_i, c_j)
    d2 = distances.calc_bonds(c_i, c_j, frag_i.dimensions)
    if not abs(d1 - d2) < tol:
        logger.debug("Shifting fragment")
        shift = (c_i - c_j) / frag_i.dimensions[:3]

        pos_j = frag_j.positions + (np.rint(shift) * frag_i.dimensions[:3])
    else:
        pos_j = frag_j.positions
    positions = np.concatenate([frag_i.positions, pos_j])
    total_ag = frag_i + frag_j
    n_atoms = len(total_ag)

    output = '{}-{}\n\n'.format(*dimer_name)
    output += 'molecular\n\n'
    output += 'geometry\n{:<15d}\n'.format(n_atoms)

    for count in range(n_atoms):
        output += (' {:<5d} {:<5s} {:>15.8f} {:>15.8f} {:>15.8f}\n'
                   ''.format(count + 1, total_ag[count].name,
                             positions[count, 0],
                             positions[count, 1],
                             positions[count, 2]))
    output += ('\n'
               'charge\n'
               '0\n\n'
               'Nonweighted\n\n'
               'dump hamiltonian\n'
               'dump overlap\n'
               'Just Matrices\n')

    return output


def run_bind(filename):
    """Run bind on a single input

    Parameters
    ----------
    frag_input : str
      string representation of yaehmop input file

    Returns
    -------
    ret : subprocess.CompletedProcess
      the return value of the subprocess call, checking not done
    """
    ret = subprocess.run('yaehmop {}'.format(filename),
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
    )
    # these are initially created with weird permissions
    os.chmod(filename + '.OV', 0o766)
    os.chmod(filename + '.HAM', 0o766)
    return ret

"""Sample ascii portion of yaehmop output:"""

"""
# ******** Extended Hueckel Parameters ********
;  FORMAT  quantum number orbital: Hii, <c1>, exponent1, <c2>, <exponent2>

ATOM: C   Atomic number: 6  # Valence Electrons: 4
	2S:   -21.4000     1.6250
	2P:   -11.4000     1.6250
ATOM: H   Atomic number: 1  # Valence Electrons: 1
	1S:   -13.6000     1.3000
ATOM: S   Atomic number: 16  # Valence Electrons: 6
	3S:   -20.0000     2.1220
	3P:   -11.0000     1.8270
ATOM: O   Atomic number: 8  # Valence Electrons: 6
	2S:   -32.3000     2.2750
	2P:   -14.8000     2.2750
ATOM: N   Atomic number: 7  # Valence Electrons: 5
	2S:   -26.0000     1.9500
	2P:   -13.4000     1.9500

; Number of orbitals
#Num_Orbitals: 1140
"""

def parse_bind_out(base, dimer_indices,
                   ag_i, ag_j,
                   keep_i=False, keep_j=False):
    """Read output from single YAEHMOP dimer calculation

    Parameters
    ----------
    base : str
      base name of files
    dimer_indices : tuple of ints
      indices of the fragments
    ag_i, ag_j : mda.AtomGroup
      AtomGroup of each fragment

    Returns
    -------
    H_mats, S_mats : dict of sparse.coo_matrix
    norbitals, nelectrons : dict of frag index to value
    """
    valence_electrons = {}
    orbitals = {}

    norbitals = {}
    nelectrons = {}

    with open(base + '.out') as f:
        line = ''
        while not line.startswith('# ******** Extended Hueckel'):
            line = next(f)

        while not line.startswith('#Num'):
            line = next(f)
            pieces = line.split()
            if not pieces:
                continue

            if pieces[0] == "ATOM:":
                atom_type = pieces[1]
                valence_electrons[atom_type] = int(pieces[-1])
                orbitals[atom_type] = 0
            else:
                for char, n in [('S', 1), ('P', 3), ('D', 5)]:
                    if char in pieces[0]:
                        orbitals[atom_type] += n

    for frag_id, ag in zip(dimer_indices, (ag_i, ag_j)):
        # number of each element
        count = Counter(ag.names)
        # number of orbitals in fragment
        norbitals[frag_id] = sum(n_e * orbitals[e]
                                 for e, n_e in count.items())
        # number of valence electrons in fragment
        nelectrons[frag_id] = sum(n_e * valence_electrons[e]
                                  for e, n_e in count.items())

    i, j = dimer_indices
    size_i = norbitals[i]
    total_size = size_i + norbitals[j]

    H_mats = {}
    S_mats = {}
    for fn, mats in ([base + '.OV', S_mats],
                     [base + '.HAM', H_mats]):
        bigmat = np.fromfile(fn)
        # matrices start with 2 ints (1 double)
        bigmat = bigmat[1:].reshape(total_size, total_size)
        # split into smaller matrices, then make sparse
        if keep_i:
            mats[i, i] = sparse.coo_matrix(bigmat[:size_i, :size_i])
        mats[i, j] = sparse.coo_matrix(bigmat[:size_i, size_i:])
        # j-i matrix is always zeros (matrix is triu)
        #mats[j, i] = sparse.coo_matrix(bigmat[size_i:, :size_i])
        if keep_j:
            mats[j, j] = sparse.coo_matrix(bigmat[size_i:, size_i:])

    return H_mats, S_mats, norbitals, nelectrons


def combine_matrices(nfrags, H_pieces, S_pieces):
    """Combine dimer matrices into entire system matrices

    Parameters
    ----------
    nfrags : int
      total number of fragments in system
    H_pieces, S_pieces : dict
      mapping of dimer_tuple to sparse matrix

    Returns
    -------
    H, S : CSR sparse arrays
      combined version of H_pieces and S_pieces
    """
    def construct_big_matrix(pieces):
        return sparse.bmat([[pieces.get((i, j), None) for j in range(nfrags)]
                            for i in range(nfrags)],
                           format='csr', dtype=np.float64)

    H = construct_big_matrix(H_pieces)
    S = construct_big_matrix(S_pieces)

    return H, S


def find_fragment_sizes(nfrags, norbitals, nelectrons):
    """Calculate offsets to index different fragments in orbital matrices

    Parameters
    ----------
    nfrags : int
      total number of fragments
    norbitals : dict
      maps fragment index to number of orbitals in the fragment
    nelectrons : dict
      maps fragment inex to number of valence electrons

    Returns
    -------
    fragsize : namedtuple
      has attributes 'starts', 'stops', 'sizes' and 'n_electrons'
      which can be indexed using fragment ids (0..nfrags-1)
    """
    sizes = np.array([norbitals.get(i, 0)
                      for i in range(nfrags)])
    starts = np.zeros(nfrags, dtype=int)
    stops = np.zeros(nfrags, dtype=int)
    n_electrons = np.array([nelectrons.get(i, 0)
                           for i in range(nfrags)])

    # find start and stop indices for each fragment
    slice_points = np.cumsum(sizes)
    starts[1:] = slice_points[:-1]
    stops[:] = slice_points[:]

    return FragSizes(
        starts=starts,
        stops=stops,
        sizes=sizes,
        n_electrons=n_electrons,
    )


def cleanup_output(base):
    """Delete output from yaehmop

    Removes:
    - input
    - .out
    - .status
    - .OV
    - .HAM

    Parameters
    ----------
    base : str
      base filename for files
    """
    os.remove(base)
    os.remove(base + '.out')
    os.remove(base + '.status')
    os.remove(base + '.OV')
    os.remove(base + '.HAM')


def run_single_dimer(ags, indices, keep_i, keep_j):
    """Create input, runs YAEHMOP and parses output

    Parameters
    ----------
    ags : tuple of mda.AtomGroup
      The dimer to run
    indices : tuple of int
      the indices of this dimer pairing, eg (4, 5)
    keep_i, keep_j : bool
      whether to keep the self contribution for fragments i & j

    Returns
    -------
    H_frag, S_frag : dict of sparse.csr_matrix
      dict of contributions for different (i,j) pairs
    orb, ele : dicts
      fragment index to orbital size and number of valence electrons
    """
    logger.debug('Calculating dimer {}'.format(indices))
    ag_i, ag_j = ags

    logger.debug("Creating yaehmop input")
    bind_input = create_bind_inp(ags, indices)
    base = '{}-{}.bind'.format(*indices)
    with open(base, 'w') as out:
        out.write(bind_input)

    logger.debug("Running yaehmop")
    completed_proc = run_bind(base)

    logger.debug("Parsing yaehmop output")
    H_frag, S_frag, orb, ele = parse_bind_out(base, indices,
                                              ag_i, ag_j,
                                              keep_i, keep_j)

    logger.debug("Removing output for {}".format(base))
    cleanup_output(base)

    return H_frag, S_frag, orb, ele


def run_all_dimers(fragments, dimers):
    """runs all dimer calculations in a parallel section

    Parameters
    ----------
    fragments : list of MDA.AtomGroup
      list of fragments
    dimers : dict
      mapping of dimer tuple to dimer AtomGroups (two fragments)

    Returns
    -------
    H_orb, S_orb, fragsize
      H_orb - csr sparse Hamiltonian matrix
      S_orb - csr sparse overlap matrix
      fragsize - namedtuple with attributes 'starts', 'stops', 'sizes' and
        'n_electrons' which can be indexed using fragment ids (0..nfrags-1)
    """
    logger.info('Starting yaehmop calculation')
    # keep track of which dimers we've seen once
    done = set()
    # coordinate representation of sparse matrices
    # ie H_coords[1, 2] gives dimer(1-2) sparse matrix
    H_coords = dict()
    S_coords = dict()
    norbitals = dict()
    nelectrons = dict()

    for (i, j), ags in tqdm(sorted(dimers.items())):
        keep_i = not i in done
        keep_j = not j in done
        done.add(i)
        done.add(j)

        H_frag, S_frag, orb, ele = run_single_dimer(ags, (i, j), keep_i, keep_j)
        H_coords.update(H_frag)
        S_coords.update(S_frag)
        norbitals.update(orb)
        nelectrons.update(ele)

    logger.info('Finding fragment sizes')
    fragsize = find_fragment_sizes(len(fragments), norbitals, nelectrons)

    logger.info('Combining matrices')
    H_orb, S_orb = combine_matrices(len(fragments), H_coords, S_coords)

    logger.info('Done with yaehmop')
    return H_orb, S_orb, fragsize
