import matplotlib.pyplot as plt
import matplotlib
import argparse
import json
import os

"""
Run in higgs-dna environment with python (not python3).
"""


# No popups with matplotlib
matplotlib.use('Agg')


if __name__ == "__main__":
    # Command-line arguments setup
    parser = argparse.ArgumentParser(description="Command line options parser", conflict_handler="resolve")

    # Input arguments
    parser.add_argument("-eff", "--eff_json", help="Efficiency JSON to load.", type=str, default=None, required=False)
    args = parser.parse_args()

    # Get current directory
    cwd = os.path.dirname(os.path.abspath(__file__))
    

    # Make for-loop with eff_json over either 1 element or the entire set of efficiency JSONs in the "jsons" directory.
    assert False, "Need to include for-loop here!"

    # Load efficiencies JSON
    with open(f"{cwd}/{eff_json}", "r") as f:
        eff_list = json.load(f)

    efficiencies = {}
    Nevents = dict.fromkeys(["initial_events", "Pre-selections", "HLT Flag", "Pre-selections + At least 4 photons", "Pre-selections + At least 4 photons + Pseudoscalar selections"])

    for mass, eff_list in eff_list.items():
        efficiencies.update({mass: Nevents.copy()})
        print(f"Mass: {mass}")

        for i, key in enumerate(Nevents.keys()):
            # Sum of genWeight should be the same for full samples!
            N = eff_list[0]
            Ni = eff_list[i]
            efficiencies[mass][key] = Ni / N * 100.0
            print('\t', f"{key:<66}", f"Numerator: {Ni:<15}", f"Denominator: {N:<20}", f"{efficiencies[mass][key]:<.2f}%")

    plt.figure()
    label_cut = 1
    for eff in list(Nevents.keys())[label_cut:]:
        x = []
        y = []
        for m in range(15,65,5):
            try:
                y.append(efficiencies[f"{m}_GeV"][eff])
                x.append(m)
            except KeyError:
                print(f"Cannot add {m} GeV mass point")
        plt.plot(x, y, "-o")
    plt.xlabel("m(a) [GeV]", fontsize=12)
    # Efficiency x Acceptance refers to acceptance (fiducial/eta cuts) and efficiency (all other selections on objects)
    plt.ylabel("Efficiency x Acceptance (%)", fontsize=12)
    plt.legend(labels=list(Nevents.keys())[label_cut:], frameon=True, loc="upper left")
    plt.grid()

    ax = plt.gca()
    ax.set_ylim(0.0, 60.0)

    # Save figure
    path = os.path.join(cwd, "efficiencies", eff_json.replace('json', 'png'))
    if not os.path.exists(os.path.join(cwd, "efficiencies")):
        os.mkdir(os.path.join(cwd, "efficiencies"))
    plt.savefig(path)
    assert os.path.isfile(path), "Saved plot does not exist."
    print(f"Saved {eff_json.replace('json', 'png')} to file.")
    plt.close()
