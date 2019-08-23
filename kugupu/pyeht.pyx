# cython: language_level=3, boundscheck=False, embedsignature=True
import cython

import numpy as np
#cimport numpy as cnp

from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport FILE, fopen, fclose, stdout
from libc.string cimport strcpy
from libc.signal cimport signal, SIGINT

#cnp.import_array()

ctypedef char BOOLEAN
ctypedef double real_t


cdef extern from "yaehmop/bind.h":
    int FAT=6
    int THIN=1
    int MOLECULAR=27
    ctypedef struct hermetian_matrix_type:
        int dim
        real_t *mat
    ctypedef struct Z_mat_type:
        pass
    ctypedef struct point_type:
        real_t x, y, z
    ctypedef struct atom_type:
        char symb[10]  # ATOM_SYMB_LEN=10
        char chg_it_vary
        int which_atom, at_number, num_valence
        point_type loc
        Z_mat_type Zmat_loc
        int ns, np, nd, nf
        real_t coul_s, coul_p, coul_d, coul_f
        real_t coeff_d1, coeff_d2, coeff_f1, coeff_f2
        real_t s_A, s_B, s_C
        real_t p_A, p_B, p_C
        real_t d_A, d_B, d_C
        real_t muller_s_E[7], muller_p_E[7], muller_d_E[7]
        real_t muller_s_Z[4], muller_p_Z[4], muller_d_Z[4]
        real_t init_s_occup, init_p_occup, init_d_occup
    ctypedef struct geom_frag_type:
        pass
    ctypedef struct Tvect_type:
        pass
    ctypedef struct xtal_defn_type:
        pass
    ctypedef struct equiv_atom_type:
        pass
    ctypedef struct cell_type:
        int dim, num_atoms, num_raw_atoms
        atom_type* atoms
        geom_frag_type* geom_frags
        char using_Zmat, using_xtal_coords
        real_t* distance_mat
        real_t num_electrons, charge
        char* sym_elems
        Tvect_type tvects[3]
        point_type recip_vects[3]
        xtal_defn_type xtal_defn
        int overlaps[3]
        point_type COM
        real_t princ_axes[3][3]
        equiv_atom_type* equiv_atoms
    ctypedef struct printing_info_type:
        pass
    ctypedef struct chg_it_parm_type:
        pass
    ctypedef struct orbital_occup_type:
        pass
    ctypedef struct k_point_type:
        point_type loc
        real_t weight
        real_t num_filled_bands
    ctypedef struct FMO_frag_type:
        pass
    ctypedef struct FMO_prop_type:
        pass
    ctypedef struct p_DOS_type:
        pass
    ctypedef struct COOP_type:
        pass
    ctypedef struct walsh_details_type:
        pass
    ctypedef struct band_info_type:
        pass
    ctypedef struct overlap_cancel_type:
        pass
    ctypedef struct detail_type:
        char title[240]
        char filename[240]
        int Execution_Mode
        BOOLEAN just_geom, avg_props, gradients, save_energies
        BOOLEAN use_symmetry, find_princ_axes
        BOOLEAN vary_zeta
        BOOLEAN eval_electrostat
        BOOLEAN weighted_Hij
        BOOLEAN dump_overlap, dump_hamil
        BOOLEAN dump_sparse_mats
        BOOLEAN dump_dist_mat
        # printing options
        int upper_level_PRT, lower_level_PRT
        real_t max_dist_PRT
        BOOLEAN distance_mat_PRT
        BOOLEAN chg_mat_PRT, Rchg_mat_PRT, wave_fn_PRT
        BOOLEAN net_chg_PRT, overlap_mat_PRT, electrostat_PRT, fermi_PRT
        BOOLEAN hamil_PRT, energies_PRT, levels_PRT
        BOOLEAN avg_OP_mat_PRT, avg_ROP_mat_PRT
        BOOLEAN no_total_DOS_PRT
        BOOLEAN just_avgE, just_matrices
        BOOLEAN mod_OP_mat_PRT, mod_ROP_mat_PRT, mod_net_chg_PRT
        BOOLEAN orbital_mapping_PRT
        int line_width
        char diag_wo_overlap
        char store_R_overlaps
        printing_info_type* step_print_options
        int num_MOs_to_print
        int *MOs_to_print
        real_t rho
        char do_chg_it
        chg_it_parm_type chg_it_parms
        real_t close_nn_contact
        int num_occup_AVG
        real_t occup_AVG_step
        int num_orbital_occups
        orbital_occup_type* orbital_occups
        int num_KPOINTS
        k_point_type* K_POINTS
        char use_automatic_kpoints, use_high_symm_p
        int points_per_axis[3]
        real_t k_offset
        int num_occup_KPOINTS
        real_t* occup_KPOINTS
        int num_bonds_OOP
        int num_FMO_frags
        int num_FCO_frags
        FMO_frag_type *FMO_frags
        FMO_prop_type *FMO_props
        int num_proj_DOS
        p_DOS_type *proj_DOS
        COOP_type *the_COOPS
        char do_moments
        int num_moments
        real_t* moments
        walsh_details_type walsh_details
        band_info_type *band_info
        int num_overlaps_off
        overlap_cancel_type *overlaps_off
        real_t the_const
        real_t sparsify_value
        real_t symm_tol
        int num_sym_ops
        real_t *characters
        char do_muller_it
        int *atoms_to_vary
        real_t muller_mix, muller_E_tol, muller_Z_tol
    # Global variables
    FILE *status_file, *output_file, *walsh_file, *band_file, *FMO_file
    FILE *MO_file
    cell_type* unit_cell
    detail_type* details
    int num_orbs
    int* orbital_lookup_table
    hermetian_matrix_type Hamil_R, Hamil_K, Overlap_R, Overlap_K
    void charge_to_num_electrons(cell_type*)

cdef extern from "yaehmop/prototypes.h":
    """#undef real"""  # conflicts with numpy real
    void fill_atomic_parms(atom_type*, int, FILE*, char*)
    void build_orbital_lookup_table(cell_type*, int*, int**)
    void run_eht(FILE*)
    void set_details_defaults(detail_type*)
    void set_cell_defaults(cell_type*)
    void cleanup_memory()


cdef void customise_details(detail_type* details):
    """settings for how we run a tight binding calculation"""
    # MOLECULAR run type
    details.Execution_Mode = MOLECULAR
    details.num_KPOINTS = 1
    details.K_POINTS = <k_point_type*>calloc(1,sizeof(k_point_type))
    details.K_POINTS[0].weight = 1.0

    # Nonweighted
    details.weighted_Hij = 0

    # Just Matrices
    details.just_matrices = 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void customise_cell(cell_type* cell, int num_atoms,
                         double[:, ::1] positions,
                         char[:, ::1] atom_types,
                         double charge):
    """Set the cell to our simulation contents

    Parameters
    ----------
    cell : ptr to cell_type
      the cell object to modify in place
    num_atoms : int
      size of simulation
    positions : double
      positions of all atoms
    atom_type : char
      names of atoms, encoded into int8 array
    charge : float
      net charge of system
    """
    cdef int i
    cdef bytes name

    # load geometry into program
    cell.num_atoms = num_atoms
    cell.atoms = <atom_type*>calloc(cell.num_atoms, sizeof(atom_type))
    for i in range(num_atoms):
        strcpy(cell.atoms[i].symb, &atom_types[i][0])
        cell.atoms[i].loc.x = positions[i][0]
        cell.atoms[i].loc.y = positions[i][1]
        cell.atoms[i].loc.z = positions[i][2]
        cell.atoms[i].which_atom = i
    cell.num_raw_atoms = cell.num_atoms

    # Figure out parameters
    fill_atomic_parms(cell.atoms, cell.num_atoms, NULL, NULL)

    # Charge 0
    cell.geom_frags = NULL
    cell.num_electrons = 0
    cell.charge = 0
    charge_to_num_electrons(cell)


cdef void steal_matrix(real_t* mat, double[::1] newmat):
    """Does something a little like dump_hermetian_mat"""
    #mat = np.array(mat)
    cdef int i, n
    n = newmat.shape[0]

    for i in range(n):
        newmat[i] = mat[i]


def run_bind(pos, elements, double charge):
    """Run tight binding calculations

    Parameters
    ----------
    positions : numpy array
      positions of atoms
    elements : iterable
      element for each atom
    charge : float
      total charge of box

    Returns
    -------
    Hamiltonian
    Overlap
    """
    cdef int num_atoms
    cdef char[:, ::1] atom_types
    cdef double[:, ::1] positions
    cdef FILE* hell  # our portal to /dev/null
    cdef double[::1] H_mat, S_mat
    # global input variables
    global unit_cell  # cell_type*
    global details  # detail_type*
    global orbital_lookup_table  # int*
    global num_orbs  # int
    # global file descriptors
    global status_file, output_file, walsh_file
    global band_file, FMO_file, MO_file
    # results arrays
    global Hamil_R, Hamil_K, Overlap_R, Overlap_K

    positions = pos.astype(np.float64, order='C')

    hell = fopen('/dev/null', 'w')
    #hell = stdout
    status_file = hell
    output_file = hell
    walsh_file = hell
    band_file = hell
    FMO_file = hell
    MO_file = hell

    num_atoms = positions.shape[0]
    # This first converts to S10 type in numpy,
    # Then casts this into int8/char type
    # TODO Add check for null terminated strings
    atom_types = np.array(elements, dtype='S10').view(
        np.int8).reshape(num_atoms, -1)

    # Allocate the input arrays and set defaults
    details = <detail_type*> calloc(1, sizeof(detail_type))
    unit_cell = <cell_type*> calloc(1, sizeof(cell_type))
    set_cell_defaults(unit_cell)
    set_details_defaults(details)

    # Change these to our data
    customise_details(details)  # TODO, add arguments and pass in here..
    customise_cell(unit_cell, num_atoms, positions, atom_types, charge)

    # Run the calculation
    build_orbital_lookup_table(unit_cell, &num_orbs, &orbital_lookup_table)
    run_eht(hell)  # it wants a file handle to use...

    # Pilfer the loot
    H_mat = np.empty(num_orbs * num_orbs, dtype=np.float64)
    S_mat = np.empty(num_orbs * num_orbs, dtype=np.float64)
    steal_matrix(Hamil_R.mat, H_mat)
    steal_matrix(Overlap_R.mat, S_mat)

    # Once we're done grabbing results, free memory again
    cleanup_memory()
    # These objects are our responsibility!!
    free(unit_cell)
    free(details)
    fclose(hell)

    return (np.asarray(H_mat).reshape(num_orbs, num_orbs),
            np.asarray(S_mat).reshape(num_orbs, num_orbs))
