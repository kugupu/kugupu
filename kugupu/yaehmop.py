"""Controls running and parsing yaehmop tight binding calculations

"""
from collections import Counter, namedtuple
from scipy import sparse
import numpy as np
import shutil
import subprocess
import os
from tqdm import tqdm
from MDAnalysis.lib import distances
import warnings

from . import logger

# expected binary to run yaehmop tight binding calculation
YAEHMOP_BIN = 'eht_bind'
if shutil.which(YAEHMOP_BIN) is None:
    warnings.warn("Could not find eht_bind in path")
    # if not found, can set yaehmop.YAEHMOP_BIN to specify

# contains information on how to index atomic orbital matrices
FragSizes = namedtuple('FragSizes',
                       ['starts', 'stops', 'sizes', 'n_electrons'])


def shift_dimer_images(frag_i, frag_j):
    """Move fragment j to be in closest image to fragment i

    Returns
    -------
    pos : numpy ndarray
      concatenated positions of frag_i and frag_j
    """
    logger.debug("Checking if fragments are in correct image")
    c_i = frag_i.center_of_mass()
    c_j = frag_j.center_of_mass()

    tol = 0.001
    d1 = distances.calc_bonds(c_i, c_j)
    d2 = distances.calc_bonds(c_i, c_j, frag_i.dimensions)
    if not abs(d1 - d2) < tol:
        logger.debug("Shifting fragment")
        shift = (c_i - c_j) / frag_i.dimensions[:3]

        pos_j = frag_j.positions + (np.rint(shift) * frag_i.dimensions[:3])
    else:
        pos_j = frag_j.positions

    return np.concatenate([frag_i.positions, pos_j])


def create_bind_input(ag, pos, name):
    """Create YAEHMOP input file

    Parameters
    ----------
    ag : mda.AtomGroup
      AtomGroup to be ran
    pos : numpy array
      positions for atoms in AtomGroup
    name : str
      name for simulation

    Returns
    -------
    output : str
      contents of Yaehmop input
      can either be piped in or written to file
    """
    logger.debug("Creating yaehmop input")
    n_atoms = len(ag)

    output = '{}\n\n'.format(name)
    output += 'molecular\n\n'
    output += 'geometry\n{:<15d}\n'.format(n_atoms)

    for count in range(n_atoms):
        output += (' {:<5d} {:<5s} {:>15.8f} {:>15.8f} {:>15.8f}\n'
                   ''.format(count + 1, ag[count].name,
                             pos[count, 0],
                             pos[count, 1],
                             pos[count, 2]))
    output += ('\n'
               'charge\n'
               '0\n\n'
               'Nonweighted\n\n'
               'dump hamiltonian\n'
               'dump overlap\n'
               'Just Matrices\n')

    return output


def run_bind(base, bind_input):
    """Run yaehmop-bind on a single input

    Parameters
    ----------
    bind_input : str
      string representation of yaehmop input file

    Returns
    -------
    ret : subprocess.CompletedProcess
      the return value of the subprocess call, checking not done
    """
    logger.debug("Writing yaehmop input")
    with open(base, 'w') as out:
        out.write(bind_input)

    logger.debug("Running yaehmop")
    ret = subprocess.run('{} {}'.format(YAEHMOP_BIN, base),
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
    )
    # these are initially created with weird permissions
    os.chmod(base + '.OV', 0o766)
    os.chmod(base + '.HAM', 0o766)
    return ret


"""Sample ascii portion of yaehmop output:
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
def parse_bind_out(fn):
    """Parse ascii section of yaehmop output

    Parameters
    ----------
    fn : str
      path to ascii yaehmop output

    Returns
    -------
    orbitals, valence_electrons : dict
      mapping of atom element to number of orbitals and valence electrons
      respectively
    """
    orbitals = {}
    valence_electrons = {}

    with open(fn, 'r') as f:
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
                orbitals[atom_type] = 0
                valence_electrons[atom_type] = int(pieces[-1])
            else:
                for char, n in [('S', 1), ('P', 3), ('D', 5)]:
                    if char in pieces[0]:
                        orbitals[atom_type] += n

    return orbitals, valence_electrons


def count_orbitals(ag, orbitals, electrons):
    """Count the number of orbitals and valence electrons in an AG

    Parameters
    ----------
    ag : AtomGroup
      AtomGroup to count orbitals and electrons for
    orbitals, electrons : dict
      mapping of element to number of orbitals/electrons

    Returns
    -------
    norbitals, nelectrons : int
      total number of orbitals and valence electrons in AtomGroup
    """
    # number of each element
    count = Counter(ag.names)
    # number of orbitals in fragment
    norbitals = sum(n_e * orbitals[e] for e, n_e in count.items())
    # number of valence electrons in fragment
    nelectrons = sum(n_e * electrons[e] for e, n_e in count.items())

    return norbitals, nelectrons


def parse_yaehmop_binary_out(fn):
    """Read yaehmop binary and return square matrix"""
    mat = np.fromfile(fn)
    # matrices start with 2 ints (1 double)
    mat = mat[1:]
    size = int(np.sqrt(mat.shape[0]))

    return mat.reshape(size, size)


def parse_yaehmop_out(base, dimer_indices,
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
    orbitals, electrons = parse_bind_out(base + '.out')

    norbitals = {}
    nelectrons = {}

    for frag_id, ag in zip(dimer_indices, (ag_i, ag_j)):
        norbitals[frag_id], nelectrons[frag_id] = count_orbitals(
            ag, orbitals, electrons)

    i, j = dimer_indices
    size_i = norbitals[i]

    H_mats = {}
    S_mats = {}
    for fn, mats in ([base + '.OV', S_mats],
                     [base + '.HAM', H_mats]):
        bigmat = parse_yaehmop_binary_out(fn)
        # split into smaller matrices, then make sparse
        if keep_i:
            mats[i, i] = sparse.coo_matrix(bigmat[:size_i, :size_i])
        mats[i, j] = sparse.coo_matrix(bigmat[:size_i, size_i:])
        # j-i matrix is always zeros (matrix is triu)
        #mats[j, i] = sparse.coo_matrix(bigmat[size_i:, :size_i])
        if keep_j:
            mats[j, j] = sparse.coo_matrix(bigmat[size_i:, size_i:])

    return H_mats, S_mats, norbitals, nelectrons


def parse_fragment_yaehmop_out(base, ag):
    """Parse output for single fragment yaehmop calculation

    Parameters
    ----------
    base : str
      base for filenames
    ag : AtomGroup
      the AtomGroup simulations were ran on

    Returns
    -------
    H_mat, S_mat : sparse.coo_matrix
      Hamiltonian and overlap on Atomic basis
    norbitals, nelectrons : int
      number of orbitals and valence electrons in system
    """
    orbitals, electrons = parse_bind_out(base + '.out')
    norbs, neles = count_orbitals(ag, orbitals, electrons)
    H_mat = sparse.coo_matrix(parse_yaehmop_binary_out(base + '.HAM'))
    S_mat = sparse.coo_matrix(parse_yaehmop_binary_out(base + '.OV'))

    return H_mat, S_mat, norbs, neles


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
    sizes = np.array([norbitals[i] for i in range(nfrags)])
    starts = np.zeros(nfrags, dtype=int)
    stops = np.zeros(nfrags, dtype=int)
    n_electrons = np.array([nelectrons[i] for i in range(nfrags)])

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
    logger.debug("Removing output for {}".format(base))

    os.remove(base)
    os.remove(base + '.out')
    os.remove(base + '.status')
    os.remove(base + '.OV')
    os.remove(base + '.HAM')


def run_single_fragment(ag, index):
    """Create input run YAEHMOP and parse output for single fragmnet

    Parameters
    ----------
    ag : mda.AtomGroup
      single fragment to run
    index : int
      identifier for this fragment

    Returns
    -------
    H_frag, S_frag : dict of sparse.csr_matrix
      dict of contributions for different (i,j) pairs
    orb, ele : dicts
      fragment index to orbital size and number of valence electrons
    """
    logger.debug('Calculating lone fragment {}'.format(index))
    base = '{}.bind'.format(index)

    bind_input = create_bind_input(ag, ag.positions, str(index))

    completed_proc = run_bind(base, bind_input)

    logger.debug('Parsing yaehmop output')
    H_mat, S_mat, norbitals, nelectrons = parse_fragment_yaehmop_out(
        base, ag)

    cleanup_output(base)

    return H_mat, S_mat, norbitals, nelectrons


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
    pos = shift_dimer_images(ag_i, ag_j)

    name = '{}-{}'.format(*indices)
    bind_input = create_bind_input(sum(ags), pos, name)
    base = name + '.bind'
    completed_proc = run_bind(base, bind_input)

    logger.debug("Parsing yaehmop output")
    H_frag, S_frag, orb, ele = parse_yaehmop_out(base, indices,
                                                 ag_i, ag_j,
                                                 keep_i, keep_j)

    cleanup_output(base)

    return H_frag, S_frag, orb, ele


def run_all_dimers(fragments, dimers):
    """Run all yaehmop calculations for dimer pairs

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
    logger.info('Starting yaehmop calculation for {} dimers'.format(len(dimers)))
    # keep track of which fragment ids we've seen once
    done = set()
    # coordinate representation of sparse matrices
    # ie H_coords[1, 2] gives dimer(1-2) sparse matrix
    H_coords = dict()
    S_coords = dict()
    # values per fragment id
    norbitals = dict()
    nelectrons = dict()

    for (i, j), ags in tqdm(sorted(dimers.items())):
        # only keep i-i contribution once
        keep_i = not i in done
        keep_j = not j in done
        done.add(i)
        done.add(j)

        H_frag, S_frag, orb, ele = run_single_dimer(
            ags, (i, j), keep_i, keep_j)
        H_coords.update(H_frag)
        S_coords.update(S_frag)
        norbitals.update(orb)
        nelectrons.update(ele)

    # for each fragment that wasn't in a dimer pairing
    for i in set(range(len(fragments))) - done:
        H_frag, S_frag, orb, ele = run_single_fragment(fragments[i], i)
        H_coords[i, i] = H_frag
        S_coords[i, i] = S_frag
        norbitals[i] = orb
        nelectrons[i] = ele

    logger.info('Finding fragment sizes')
    fragsize = find_fragment_sizes(len(fragments), norbitals, nelectrons)

    logger.info('Done with yaehmop')
    return H_coords, S_coords, fragsize
