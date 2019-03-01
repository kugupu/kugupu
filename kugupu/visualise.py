"""Visualise results"""
import MDAnalysis as mda
import numpy as np
import warnings

COL_DICT = {'r': [1, 0, 0],
            'g': [0, 1, 0],
            'b': [0, 0, 1]}

def move_image(ref_point, atomgroup):
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


def gen_frag_positions(fragments):
    return np.array([frag.center_of_geometry() for frag in fragments])


def draw_fragment_centers(view, fragments, color='r'):
    """Draw fragment centers onto nglview View"""
    pos = gen_frag_positions(fragments).flatten().tolist()

    view.shape.add_buffer("sphere",
                         position=pos,
                         color=COL_DICT[color] * len(fragments),
                         radius=[1.5] * len(fragments))

def draw_fragment_links(view, fragments, links, color='r'):
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


def gather_network(g):
    """Move contents of g to same image

    Will pick centermost fragment in box as start point,
    then spread outwards moving joined fragments to

    Parameters
    ----------
    g : nx.Graph
      graph with AtomGroups representing fragments as nodes
    """
    frags = list(g.nodes())

    box = frags[0].dimensions
    center = box[:3] / 2.
    # find fragment most near the center?
    center_frag = np.argmin(
        mda.lib.distances.distance_array(
            gen_frag_positions(frags),
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
            move_image(ref_point, neb)
            starts.add(neb)

        done.add(center)


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
    frags = list(network.nodes())
    if not isinstance(mda.coordinates.memory.MemoryReader,
                      frags[0].universe.trajectory):
        warnings.warn("Visualisation works best with in-memory trajectory. "
                      "Consider using transfer_to_memory")
    import nglview as nv

    # move contents of network into primary unit cell
    gather_network(network)

    if view is None:
        view = nv.NGLWidget()

    if show_molecules:
        view.add_trajectory(sum(frags).select_atoms('prop mass > 2.0'))

    #view.clear_representations()
    #view.add_ball_and_stick(opacity=0.5)

    draw_fragment_centers(view, frags, color=color)
    draw_fragment_links(view, frags, network.edges(), color=color)

    #view.add_unitcell()

    return view
