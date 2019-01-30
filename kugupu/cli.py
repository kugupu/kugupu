"""Cli goes here"""

def cli_kugupu(dcdfile, topologyfile, param_file):
    """Command line entry to Kugupu

    Parameters
    ----------
    dcdfile, topologyfile : str
      inputs to MDAnalysis
    param_file : str
      filename which holds run settings
    """
    #creates universe object from trajectory
    u = make_universe(topologyfile, dcdfile)
    # returns parameter dictionary from parameter yaml file
    params = read_param_file(param_file)

    hams = generate_traj_H_frag(u, **params)

    # collects output from entire trajectory into a pandas dataframe
    dataframe = run_analysis(H_frag, networks)

    write_shizznizz(dataframe)
