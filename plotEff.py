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
    
    if not os.path.exists(os.path.join(cwd, "tables")):
        os.mkdir(os.path.join(cwd, "tables"))

    # Load efficiencies JSON
    with open(f"{cwd}/jsons/{args.eff_json}", "r") as f:
        eff_list = json.load(f)

    with open(os.path.join(cwd, "tables", args.eff_json.replace(".json", ".txt")), "w") as f:
        f.write(f"Inputs: {args.eff_json.replace('.png', '')}\n\n")

    efficiencies = {}
    Nevents = dict.fromkeys(["initial_events", "Pre-selections", "HLT Flag", "Pre-selections + At least 4 photons", "Pre-selections + At least 4 photons + Pseudoscalar selections"])

    for mass, eff_list in eff_list.items():
        efficiencies.update({mass: Nevents.copy()})
        with open(os.path.join(cwd, "tables", args.eff_json.replace(".json", ".txt")), "a") as f:
            print(f"Mass: {mass}")
            f.write(f"Mass: {mass}\n")

        for i, key in enumerate(Nevents.keys()):
            # Sum of genWeight should be the same for full samples!
            N = eff_list[0]
            Ni = eff_list[i]
            efficiencies[mass][key] = Ni / N * 100.0
            out_str = '\t', f"{key:<66}", f"Numerator: {Ni:<15}", f"Denominator: {N:<20}", f"{efficiencies[mass][key]:<.2f}%"
            print("\t".join(out_str))
            with open(os.path.join(cwd, "tables", args.eff_json.replace(".json", ".txt")), "a") as f:
                f.write("\t".join(out_str) + '\n')

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
    path = os.path.join(cwd, "efficiencies", args.eff_json.replace('json', 'png'))
    if not os.path.exists(os.path.join(cwd, "efficiencies")):
        os.mkdir(os.path.join(cwd, "efficiencies"))
    plt.savefig(path)
    assert os.path.isfile(path), "Saved plot does not exist."
    print(f"Saved {args.eff_json.replace('json', 'png')} to file.")
    plt.close()
