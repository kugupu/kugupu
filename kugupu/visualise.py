"""Visualise results"""
import io
import MDAnalysis as mda
import networkx as nx
import numpy as np
import warnings

COL_DICT = {'r': [1, 0, 0],
            'g': [0, 1, 0],
            'b': [0, 0, 1]}


def _copy_universe(atomgroup):
    """Make a new Universe containing only atomgroup

    Useful for giving to nglview as the coordinates are safe from modification
    """
    f = mda.lib.util.NamedStream(io.StringIO(), 'tmp.pdb')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # write out atomgroup to a stream
        with mda.Writer(f, n_atoms=len(atomgroup)) as w:
            w.write(atomgroup)
    # read stream back in as universe
    f.seek(0)
    u_new = mda.Universe(f)

    u_new.transfer_to_memory()

    return u_new


def _move_image(ref_point, atomgroup):
    """Move atomgroup to closest image to ref_point

    Parameters
    ----------
    ref_point : coordinate
      position to move atomgroup close to
    atomgroup : mda.AtomGroup
      atomgroup will be translated in place

    Returns
    -------
    shift : ndarray (3,)
      the applied shift expressed as multiples of box dimensions
    """
    vec = ref_point - atomgroup.center_of_geometry()

    shift = np.rint((ref_point - atomgroup.center_of_geometry()) / atomgroup.dimensions[:3])

    if shift.any():
        atomgroup.translate(shift * atomgroup.dimensions[:3])

    return shift


def _gen_frag_positions(fragments):
    return np.array([frag.center_of_geometry() for frag in fragments])


def _draw_fragment_centers(view, fragments, color='r'):
    """Draw fragment centers onto nglview View"""
    pos = _gen_frag_positions(fragments).flatten().tolist()

    view.shape.add_buffer("sphere",
                         position=pos,
                         color=COL_DICT[color] * len(fragments),
                         radius=[1.5] * len(fragments))

def _draw_fragment_links(view, fragments, links, color='r'):
    """Draw links between fragment centers"""
    p1, p2 = [], []
    for i, j in links:
        p1 += i.center_of_geometry().tolist()
        p2 += j.center_of_geometry().tolist()

    view.shape.add_buffer("cylinder",
                          position1=p1,
                          position2=p2,
                          color=COL_DICT[color] * len(links),
                          radius=[0.75] * len(links))


def gather_network(frags):
    """Move contents of g to same image

    Will pick centermost fragment in box as start point,
    then spread outwards moving joined fragments to

    Parameters
    ----------
    g : nx.Graph
      graph with AtomGroups representing fragments as nodes
    """
    for f in frags:
        mda.lib.mdamath.make_whole(f)
    # generate fragment positions
    pos = _gen_frag_positions(frags)
    distmat = mda.lib.distances.distance_array(
        pos, pos, box=frags[0].dimensions)
    # make self contribution infinite
    distmat[np.diag_indices_from(distmat)] = np.inf

    # time to make graph
    g = nx.Graph()
    g.add_nodes_from(frags)
    for x, y, _ in nx.minimum_spanning_edges(nx.Graph(distmat)):
        g.add_edge(frags[x], frags[y])

    box = frags[0].dimensions
    center = box[:3] / 2.
    # find fragment most near the center?
    center_frag = np.argmin(
        mda.lib.distances.distance_array(
            pos,
            center
        )
    )
    starts = set([frags[center_frag]])
    done = set()

    while len(done) < len(frags):
        center = list(starts - done)[0]
        ref_point = center.center_of_geometry()
        for neb in g[center]:
            if neb in starts:
                continue
            _move_image(ref_point, neb)
            starts.add(neb)

        done.add(center)

    return g


def draw_fragments(*fragments):
    """Draw many fragments

    Will make molecules whole and choose the image of each
    which makes them as close as possible

    Parameters
    ----------
    fragments : list of AtomGroup
      the thing to draw
    """
    import nglview as nv

    u = _copy_universe(sum(fragments))

    # gather network into same image
    g = gather_network(u.atoms.fragments)

    v = nv.show_mdanalysis(sum(g.nodes()))

    return v


def draw_network(network, view=None, color='r',
                 show_molecules=True):
    """Draw a coupling network with nglview in a Jupyter notebook

    Parameters
    ----------
    network : networkx.Graph
      molecular graph to show
    view : nglview.NGLWidget, optional
      viewer to draw molecule onto.  If None a new one
      will be created and returned
    color : str, optional
      color to draw the network in, default red
    show_molecules : bool, optional
      whether the show the atomic representation (default True)

    Returns
    -------
    view : nglview.NGLWidget
      the nglview object showing the molecule
    """
    import nglview as nv

    u = _copy_universe(sum(network.nodes()))

    # move contents of network into primary unit cell
    g = gather_network(u.atoms.fragments)

    if view is None:
        view = nv.NGLWidget()

    # nglview throws a lot of useless warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if show_molecules:
            view.add_trajectory(u.select_atoms('prop mass > 2.0'))

        #view.clear_representations()
        #view.add_ball_and_stick(opacity=0.5)

        #_draw_fragment_centers(view, frags, color=color)
        #_draw_fragment_links(view, frags, network.edges(), color=color)

    #view.add_unitcell()

    return view
