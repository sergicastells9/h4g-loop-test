import uproot
import awkward as ak
import numpy as np
import itertools
import more_itertools
import vector
import argparse
import os
import json
import time


"""
Run with `python run_h4g.py`.
Use the higgs-dna conda environment.
"""

# EA values for Run3 from Egamma
EA1_EB1 = 0.102056
EA2_EB1 = -0.000398112
EA1_EB2 = 0.0820317
EA2_EB2 = -0.000286224
EA1_EE1 = 0.0564915
EA2_EE1 = -0.000248591
EA1_EE2 = 0.0428606
EA2_EE2 = -0.000171541
EA1_EE3 = 0.0395282
EA2_EE3 = -0.000121398
EA1_EE4 = 0.0369761
EA2_EE4 = -8.10369e-05
EA1_EE5 = 0.0369417
EA2_EE5 = -2.76885e-05


# Apply pfPhoIsoCorrection
def phoIsoCorrection(
    eta,
    iso,
    rho,
    year
) -> ak.Array:
    
    photon_abs_eta = np.abs(eta)

    if year != 2018:
        # EB regions
        region_EB_0 = (photon_abs_eta > 0.0) and (photon_abs_eta < 1.0)
        rhoCorr_EB_0 = iso - (rho * EA1_EB1) - (rho * rho * EA2_EB1)

        region_EB_1 = (photon_abs_eta > 1.0) and (photon_abs_eta < 1.4442)
        rhoCorr_EB_1 = iso - (rho * EA1_EB2) - (rho * rho * EA2_EB2)

        # EE regions
        region_EE_0 = (photon_abs_eta > 1.566) and (photon_abs_eta < 2.0)
        rhoCorr_EE_0 = iso - (rho * EA1_EE1) - (rho * rho * EA2_EE1)

        region_EE_1 = (photon_abs_eta > 2.0) and (photon_abs_eta < 2.2)
        rhoCorr_EE_1 = iso - (rho * EA1_EE2) - (rho * rho * EA2_EE2)

        region_EE_2 = (photon_abs_eta > 2.2) and (photon_abs_eta < 2.3)
        rhoCorr_EE_2 = iso - (rho * EA1_EE3) - (rho * rho * EA2_EE3)

        region_EE_3 = (photon_abs_eta > 2.3) and (photon_abs_eta < 2.4)
        rhoCorr_EE_3 = iso - (rho * EA1_EE4) - (rho * rho * EA2_EE4)

        region_EE_4 = (photon_abs_eta > 2.4) and (photon_abs_eta < 2.5)
        rhoCorr_EE_4 = iso - (rho * EA1_EE5) - (rho * rho * EA2_EE5)

        if region_EB_0:
            return rhoCorr_EB_0
        elif region_EB_1:
            return rhoCorr_EB_1
        elif region_EE_0:
            return rhoCorr_EE_0
        elif region_EE_1:
            return rhoCorr_EE_1
        elif region_EE_2:
            return rhoCorr_EE_2
        elif region_EE_3:
            return rhoCorr_EE_3
        elif region_EE_4:
            return rhoCorr_EE_4
        if not region_EB_0 and not region_EB_1 and not region_EE_0 and not region_EE_1 and not region_EE_2 and not region_EE_3 and not region_EE_4:
            return iso

    else:
        if photon_abs_eta < 1.5:
            return iso - rho * 0.16544
        elif photon_abs_eta > 1.5:
            return iso - rho * 0.13212


###   Start selections functions   ###
def doSelections(
    initial_events,
    triggers,
    pre_sel,
    four_photons,
    pseudos,
    event,
    apply_m55_trigger,
    year
) -> tuple:
    
    ### Pre-selections ###

    # Minimum 2 photons cut
    if len(event) < 2:
        return (initial_events, triggers, pre_sel, four_photons, pseudos, event)

    # Get descending photon pT
    descending_pt = ak.argsort(event["Photon_pt"], ascending=False)

    # Format rho to size of photons array
    rho = event["Rho_fixedGridRhoAll" if year != 2018 else 'fixedGridRhoFastjetAll'] * np.ones_like(event["Photon_pt"])
    event["Photon_pfPhoIso03_rhoCorrected"] = [None] * len(descending_pt)

    # Keep good photon indices for diphoton cuts
    good_photons = []

    for i in descending_pt:
        ## Fiducial cuts
        #fiducial = np.abs(event["Photon_eta"][i]) < max_eta and (np.abs(event["Photon_eta"][i]) > gap_endcap_eta or np.abs(event["Photon_eta"][i]) < gap_barrel_eta)
        fiducial = event["Photon_isScEtaEB"][i] or event["Photon_isScEtaEE"][i]
        
        ## Trigger mimicking cuts
        pt = True
        if i == 0:
            pt = event["Photon_pt"][i] > 30.0
        elif i == 1:
            pt = event["Photon_pt"][i] > 18.0
        hoe = event["Photon_hoe"][i] < 0.08
        
        sigma_ieie = r9 = False
        if event["Photon_isScEtaEB"]:
            # EB
            sigma_ieie = event["Photon_sieie"][i] < 0.015
            r9 = event["Photon_r9"][i] > 0.5
        elif event["Photon_isScEtaEE"]:
            # EE
            sigma_ieie = event["Photon_sieie"][i] < 0.035
            r9 = event["Photon_r9"][i] > 0.8

        event["Photon_pfPhoIso03_rhoCorrected"][i] = phoIsoCorrection(event["Photon_eta"][i], event["Photon_pfPhoIso03"][i], rho[i], year)
        pfPhoIso = event["Photon_pfPhoIso03_rhoCorrected"][i] < 4.0
        trackerIso = event["Photon_trkSumPtHollowConeDR03"][i] < 6.0
        electron_veto = event["Photon_pixelSeed"][i] == False
        
        hlt = pt and hoe and pfPhoIso and trackerIso and electron_veto and sigma_ieie and r9

        ## miniAOD cuts
        mini_r9 = event["Photon_r9"][i] > 0.8
        chad_iso = event['Photon_pfRelIso03_chg_quadratic' if args.year != 2018 else 'Photon_pfRelIso03_chg'][i] < 20.0
        chad_iso_pt = event['Photon_pfRelIso03_chg_quadratic' if args.year != 2018 else 'Photon_pfRelIso03_chg'][i] * event["Photon_pt"][i] < 0.3
        mini = mini_r9 or chad_iso or chad_iso_pt

        if fiducial and hlt and mini:
            good_photons.append(i)

    # Minimum 2 photons cut for diphotons
    if len(good_photons) < 2:
        return (initial_events, triggers, pre_sel, four_photons, pseudos, event)

    best_dipho = 0.0
    diphoton = None
    for perm in set(more_itertools.distinct_combinations(good_photons, 2)):
        pho1_vec = vector.obj(pt=event["Photon_pt"][perm[0]], eta=event["Photon_eta"][perm[0]], phi=event["Photon_phi"][perm[0]], mass=0.0)
        pho2_vec = vector.obj(pt=event["Photon_pt"][perm[1]], eta=event["Photon_eta"][perm[1]], phi=event["Photon_phi"][perm[1]], mass=0.0)
        assert pho1_vec.pt > pho2_vec.pt

        diphoton_vec = pho1_vec + pho2_vec

        if diphoton_vec.mass != 0:
            if pho1_vec.pt > 30.0 and pho2_vec.pt > 18.0 and pho1_vec.pt / diphoton_vec.mass > 30.55/65.0 and pho2_vec.pt / diphoton_vec.mass > 18.20/65.0:
                if not apply_m55_trigger:
                    if pho1_vec.pt**2 + pho2_vec.pt**2 > best_dipho:
                        best_dipho = pho1_vec.pt**2 + pho2_vec.pt**2
                        diphoton = (diphoton_vec, pho1_vec, pho2_vec, perm)
                else:
                    if diphoton_vec.mass > 55.0:
                        if pho1_vec.pt**2 + pho2_vec.pt**2 > best_dipho:
                            best_dipho = pho1_vec.pt**2 + pho2_vec.pt**2
                            diphoton = (diphoton_vec, pho1_vec, pho2_vec, perm)

    if diphoton is not None:
        # Add diphoton variables to events array
        event["dipho_mass"] = diphoton[0].mass
        event["pT1_m_gg"] = diphoton[1].pt / diphoton[0].mass
        event["pT2_m_gg"] = diphoton[2].pt / diphoton[0].mass
        event["lead_pt"] = diphoton[1].pt
        event["sublead_pt"] = diphoton[2].pt
        event["lead_isScEtaEB"] = event["Photon_isScEtaEB"][diphoton[3][0]]
        event["lead_isScEtaEE"] = event["Photon_isScEtaEE"][diphoton[3][0]]
        event["sublead_isScEtaEB"] = event["Photon_isScEtaEB"][diphoton[3][1]]
        event["sublead_isScEtaEE"] = event["Photon_isScEtaEE"][diphoton[3][1]]

        pre_sel.append(event)

    else:
        return (initial_events, triggers, pre_sel, four_photons, pseudos, event)

    
    ### HLT Flag Check ###

    if event['HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId' if year != 2018 else 'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto'] == True:
        triggers.append(event)
    else:
        return (initial_events, triggers, pre_sel, four_photons, pseudos, event)

    
    ### 4 Photon Cut ###

    if len(event["Photon_pt"]) >= 4:
        four_photons.append(event)
    else:
        return (initial_events, triggers, pre_sel, four_photons, pseudos, event)


    ### Pseudoscalar Cuts ###

    if len(event["Photon_pt"]) < 4:
        assert len(event["Photon_pt"]) >= 4, "Fewer than 4 photons remaining..."
        return (initial_events, triggers, pre_sel, four_photons, pseudos, event)

    # Get descending photon pT and keep only first 4
    descending_pt = ak.argsort(event["Photon_pt"], ascending=False)
    for key in event.keys():
        #if "fixedGridRhoAll" not in key and "fixedGridRhoFastjetAll" not in key and "HLT" not in key and type(event[key]) is list:
        if type(event[key]) is list:
            event[key] = [event[key][i] for i in descending_pt if i < 4]

    assert len(event["Photon_pt"]) == 4
    descending_pt = ak.argsort(event["Photon_pt"], ascending=False)

    good_photons = []
    for i in descending_pt:
        ## Fiducial cuts
        fiducial = event["Photon_isScEtaEB"][i] or event["Photon_isScEtaEE"][i]
        
        ## pT cuts
        pt = True
        if i == 0:
            pt = event["Photon_pt"][i] > 30.0
        elif i == 1:
            pt = event["Photon_pt"][i] > 18.0
        else:
            pt = event["Photon_pt"][i] > 15.0

        if fiducial and pt:
            good_photons.append(i)

    # Minimum 4 photons cut for pseudoscalars
    if len(good_photons) < 4:
        return (initial_events, triggers, pre_sel, four_photons, pseudos, event)

    # Make photon vectors
    mixes = []
    pseudoscalars = []
    perms_list = [[1,2,3,4], [1,3,2,4], [1,4,2,3]]
    for perm_abs in perms_list:
        perm = [p - 1 for p in perm_abs]
        pho1_vec = vector.obj(pt=event["Photon_pt"][perm[0]], eta=event["Photon_eta"][perm[0]], phi=event["Photon_phi"][perm[0]], mass=0.0)
        pho2_vec = vector.obj(pt=event["Photon_pt"][perm[1]], eta=event["Photon_eta"][perm[1]], phi=event["Photon_phi"][perm[1]], mass=0.0)
        pho3_vec = vector.obj(pt=event["Photon_pt"][perm[2]], eta=event["Photon_eta"][perm[2]], phi=event["Photon_phi"][perm[2]], mass=0.0)
        pho4_vec = vector.obj(pt=event["Photon_pt"][perm[3]], eta=event["Photon_eta"][perm[3]], phi=event["Photon_phi"][perm[3]], mass=0.0)
        mixes.append([pho1_vec, pho2_vec, pho3_vec, pho4_vec])

    # Mass mixing
    best_perm = None
    best_dM = -1
    assert len(mixes) == 3 
    for i, mix in enumerate(mixes):
        dM = np.abs((mix[0] + mix[1]).mass - (mix[2] + mix[3]).mass)
        if dM < best_dM or best_dM == -1:
            best_dM = dM
            best_perm = i

    assert type(best_perm) is int, f"{best_perm} {best_dM} {len(mixes)}"

    best_mix = mixes[best_perm]
    event["dM"] = best_dM

    # Make pseudoscalars and keep leading/subleading photons from each pseudoscalar
    order = None
    try:
        pseudo1 = best_mix[0] + best_mix[1]
        pseudo2 = best_mix[2] + best_mix[3]
        assert pseudo1.pt > pseudo2.pt
        order = (best_mix[0], best_mix[1], best_mix[2], best_mix[3])
    except AssertionError:
        pseudo2 = best_mix[0] + best_mix[1]
        pseudo1 = best_mix[2] + best_mix[3]
        assert pseudo1.pt > pseudo2.pt
        order = (best_mix[2], best_mix[3], best_mix[0], best_mix[1])
    assert order[0].pt > order[1].pt and order[2].pt and order[3].pt

    higgs = pho1_vec + pho2_vec + pho3_vec + pho4_vec
    assert (pseudo1 + pseudo2).mass - higgs.mass < 0.0001, f"{(pseudo1 + pseudo2).mass} {higgs.mass}"

    # Higgs mass search window
    if higgs.mass > 110.0 and higgs.mass < 180.0:
        pseudoscalars.append((pseudo1, pseudo2))

    # Add leading/subleading pT and pT/ma to event
    event["LeadPs_mass"] = pseudo1.mass
    event["SubleadPs_mass"] = pseudo2.mass
    event["LeadPs_leading_pho_pt"] = best_mix[0].pt
    event["LeadPs_subleading_pho_pt"] = best_mix[1].pt
    event["SubleadPs_leading_pho_pt"] = best_mix[2].pt
    event["SubleadPs_subleading_pho_pt"] = best_mix[3].pt
    event["pT1_ma1"] = event["LeadPs_leading_pho_pt"] / event["LeadPs_mass"] if event["LeadPs_mass"] != 0 else None
    event["pT2_ma1"] = event["LeadPs_subleading_pho_pt"] / event["LeadPs_mass"] if event["LeadPs_mass"] != 0 else None
    event["pT1_ma2"] = event["SubleadPs_leading_pho_pt"] / event["SubleadPs_mass"] if event["SubleadPs_mass"] != 0 else None
    event["pT2_ma2"] = event["SubleadPs_subleading_pho_pt"] / event["SubleadPs_mass"] if event["SubleadPs_mass"] != 0 else None
    event["mass_gggg"] = higgs.mass

    assert len(pseudoscalars) <= 1, f"{len(pseudoscalars)}"
    if len(pseudoscalars) == 1:
        pseudos.append(event)

    return (initial_events, triggers, pre_sel, four_photons, pseudos, event)



###   Event Loop Function   ###
def loopEvents(
    initial_events,
    triggers,
    pre_sel,
    four_photons,
    pseudos,
    events,
    apply_m55_trigger,
    year
) -> tuple:

    milestones = [1, 10, 25, 50, 75, 100]
    for i, ak_event in enumerate(events):
        event = ak.to_list(ak_event)

        # Print at intervals to show progress
        percentage_complete = (100.0 * (i+1) / len(events))
        if percentage_complete >= milestones[0]:
            print(f"Progress at {milestones[0]}%!")
            milestones = milestones[1:]
            #break
 
        # Always add event to initial events array
        initial_events.append(event)

        # Do selections
        initial_events, triggers, pre_sel, four_photons, pseudos, event = doSelections(
            initial_events,
            triggers,
            pre_sel,
            four_photons,
            pseudos,
            event,
            apply_m55_trigger,
            args.year
        )

    return (initial_events, triggers, pre_sel, four_photons, pseudos, len(initial_events) / len(events) * 100)



###   Main section   ###
if __name__ == "__main__":
    # Command-line arguments setup
    parser = argparse.ArgumentParser(description="Command line options parser", conflict_handler="resolve")

    # Input arguments
    parser.add_argument("-m", "--mass", help="Choose mass point to run. Defaults to all.", type=int, default=None, required=False)
    parser.add_argument("-m55", "--apply_m55_trigger", help="Apply m_gg > 55 GeV trigger and preselection.", action="store_true", required=False)
    parser.add_argument("-y", "--year", help="Year of samples to use. Defaults to 2022.", type=int, default=2022, required=False)
    args = parser.parse_args()

    # Get current directory
    cwd = os.path.dirname(os.path.abspath(__file__))

    # Set up masses for loop
    masses = []
    if args.mass is None:
        masses = [m for m in range(15,65,5)]
    else:
        assert os.path.exists(f"{cwd}/root_samples/{args.year}/{args.mass}_GeV.root")
        masses.append(args.mass)

    # Create blank efficiency JSON if it doesn't exist to start clean
    eff = dict.fromkeys([f"{m}_GeV" for m in range(15,65,5)])
    eff_path = f"efficiencies_{args.year}.json" if not args.apply_m55_trigger else f"efficiencies_m55_{args.year}.json"
    if not os.path.exists(eff_path):
        with open(eff_path, "w") as f:
            json.dump(eff, f, indent=4)
    
    # Calculate efficiency for chosen masses
    for mass in masses:
        # Load Events tree from ROOT file using numpy arrays
        events_ak = uproot.open(f"{cwd}/root_samples/{args.year}/{mass}_GeV.root:Events").arrays([
            'Photon_electronVeto',
            'Photon_isScEtaEB',
            'Photon_isScEtaEE',
            'Photon_pixelSeed',
            'Photon_hoe',
            'Photon_mvaID',
            'Photon_pfPhoIso03',
            'Photon_pfRelIso03_chg_quadratic' if args.year != 2018 else 'Photon_pfRelIso03_chg',
            'Photon_pt',
            'Photon_eta',
            'Photon_phi',
            'Photon_r9',
            'Photon_sieie',
            'Photon_trkSumPtHollowConeDR03',
            'Rho_fixedGridRhoAll' if args.year != 2018 else 'fixedGridRhoFastjetAll',
            'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId' if args.year != 2018 else 'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto',
            'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_Mass55' if args.year != 2018 else 'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto_Mass55']
        )
        print(f"Loaded one file for m_a = {mass} GeV!")

        # Setup for efficiencies at selection steps
        Nevents = {
            "initial_events": (0, []),
            "pre_selections": (0, []),
            "trigger_flag": (0, []),
            "pre_selections+4photon": (0, []),
            "selections": (0, []),
        }

        # Loop through events, apply cuts, and keep events passing cuts at each stage
        print(f"Starting event loop for {len(events_ak)} events...")
        if args.apply_m55_trigger:
            print("Applying m_gg > 55 pre-selection.")
        outputs = loopEvents(
            Nevents["initial_events"][1],
            Nevents["pre_selections"][1],
            Nevents["trigger_flag"][1],
            Nevents["pre_selections+4photon"][1],
            Nevents["selections"][1],
            events_ak,
            args.apply_m55_trigger,
            args.year
        )

        print(f"Finished processing {len(outputs[0])} events! {outputs[5]}% of total.\n")

        # Save selected arrays to Nevents and count # of events remaining at each step
        Nevents["initial_events"] = (len(outputs[0]), outputs[0])
        Nevents["trigger_flag"] = (len(outputs[1]), outputs[1])
        Nevents["pre_selections"] = (len(outputs[2]), outputs[2])
        Nevents["pre_selections+4photon"] = (len(outputs[3]), outputs[3])
        Nevents["selections"] = (len(outputs[4]), outputs[4])

        # Print efficiency at each selection step
        selection_eff_str = f"~~~  Selection Efficiency for m_a = {mass} GeV  ~~~"
        print(selection_eff_str)
        for key, value in Nevents.items():
            print(f"{key:<25}", f"Numerator: {value[0]:<15}", f"Denominator: {Nevents['initial_events'][0]:<20}", f"Efficiency: {value[0] / Nevents['initial_events'][0] * 100:<.2f}%")

        # Save efficiency for given mass point to JSON        
        with open(eff_path, "r+") as f:
            json_in = json.load(f)

			# Store efficiency for this mass point
            json_in.update({f"{mass}_GeV": [
				Nevents["initial_events"][0],
				Nevents["pre_selections"][0],
				Nevents["trigger_flag"][0],
				Nevents["pre_selections+4photon"][0],
				Nevents["selections"][0]
			]})

            # Clear file contents before writing to ensure only one efficiency dictionary remains
        with open(eff_path, "w") as f:
            json.dump(json_in, f, indent=4)

        # Save remaining to parquet for plotting
        percentage = "" if outputs[5] == 100 else f"_{outputs[5]:.0f}pc"
        if not os.path.exists("outputs"):
            os.mkdir("outputs")

        current_day = time.strftime("%Y%m%d", time.localtime())
        parquet_path = f"outputs/{mass}_GeV{percentage}_{current_day}.parquet"
        df = ak.to_pandas(Nevents["selections"][1])
        df.to_parquet(parquet_path)

        print()
        assert os.path.exists(parquet_path)
        if percentage == "":
            print(f"Saved one {mass} GeV sample file to parquet.")
        else:
            print(f"Saved {percentage.replace('_','')}% of one {mass} GeV sample file to parquet.")

        print("~" * len(selection_eff_str))
