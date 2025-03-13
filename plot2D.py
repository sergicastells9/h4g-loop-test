import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import argparse
from ROOT import kBlack, kRed, kGreen, kOrange, kBlue, TCanvas, TH2D, TLegend, TH1D, TLatex, gPad, gROOT, TPaveText, gStyle

"""
Run this in root6 environment with python (not python3).
"""



# Set batch so no plots pop up
gROOT.SetBatch(True)


def fillHist(
    branch: str,
    sample: pd.DataFrame,
    bounds: List[float],
    name: Optional[str] = None,
    binsScale: Optional[float] = 1.0,
    normalize: Optional[bool] = True,
    customWeights: Optional[pd.DataFrame] = None
) -> TH1D:
    """
    Fills histogram with data from samples dataframe. Binning and edges are based on bounds input.
    """

    # Set name to branch if not provided
    if name is None:
        name = branch

    # Histogram with custom number of equal bins, bounded by (bounds[0], bounds[1])
    hist = TH1D(name, branch, int((bounds[1]-bounds[0])*binsScale), bounds[0], bounds[1])
    branch_df = sample[branch].tolist()

    try:
        weights = np.ones_like(sample["genWeight"].tolist())
        print("Using genWeight for weights.")
    except KeyError as err:
        weights = np.ones_like(branch_df)
        print("Using 1.0 for weights.")

    if customWeights is not None:
        weights = customWeights.tolist()
        print("Using additional custom weights.")

    # Fill histogram
    for data, weight in zip(branch_df, weights):
        if type(data) is np.ndarray:
            data2 = data.tolist()
            weights2 = [weight] * len(data2)
            for d2, w2 in zip(data2, weights2):
                hist.Fill(d2, w2)
        else:
            hist.Fill(data, weight)

    # Normalize
    if normalize:
        if hist.Integral("width") > 0.0:
            hist.Scale(1.0 / hist.Integral())
        else:
            hist.Scale(1.0)

    # Turn of stat box by default
    gStyle.SetOptStat(0)

    assert hist.GetEntries() != 0, f"No entries in {branch} histogram within bounds! {sample[branch]}"
    note = f"Filled {branch} histogram with {int(hist.GetEntries())} events."
    print(f"{note:<50}\t Bounds: [{bounds[0]},{bounds[1]}]\t NBins: {int((bounds[1]-bounds[0])*binsScale)}")

    return hist


def loadSamples(
    directory: str,
    branches: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Scans directory (or a single file) and reads in parquet files. Creates list of dataframes with parquet file contents and merges them.
    """

    assert os.path.exists(directory), f"Directory ({directory}) does not exist."

    samples_files = []
    # Load from file
    try:
        samples_files.append(pd.read_parquet(directory, columns=branches))
    except ArrowInvalid as err:
        print(f"Could not load {name}")
        raise err

    # Fill df with all file contents
    assert len(samples_files) != 0, f"Samples size is 0!\t {directory}"
    df = pd.concat(samples_files, axis=0, ignore_index=True)

    assert df.shape[0] != 0, "Number of events = 0"
    print(f"Loaded {len(samples_files)} parquet file(s) with {len(df)} events.")

    return df


def plotFancy(
    canvas: TCanvas,
    title: str,
    prelim: bool,
    lumiTxt: Optional[str] = "",
    sub: Optional[str] = None,
    inPlot: Optional[bool] = True,
    pad: Optional[bool] = False,
    adjust_sub_fit: Optional[bool] = False,
    colz: bool = False
) -> None:
    """
    Sets up CMS style canvas. Tested with 1000x1000 canvas.
    """

    # Select canvas
    canvas.cd()

    # General canvas settings
    gPad.SetFillColor(0)
    gPad.SetBorderMode(0)
    gPad.SetBorderSize(10)
    gPad.SetTickx(1)
    gPad.SetTicky(1)
    gPad.SetFrameFillStyle(0)
    gPad.SetFrameLineStyle(0)
    # gPad.SetFrameLineWidth(3)
    gPad.SetFrameBorderMode(0)
    gPad.SetFrameBorderSize(10)

    if not pad:
        canvas.SetLeftMargin(0.16)
        canvas.SetRightMargin(0.05)
        canvas.SetBottomMargin(0.14)
    if colz:
        canvas.SetRightMargin(0.15)

    # Latex settings
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextFont(43)
    latex.SetTextSize(20)
    latex.SetTextAlign(31)
    latex.SetTextAlign(11)

    low = {"lumi": [], "cms": [], "title": [], "prelim": []}
    textSize = {"lumi": 0, "cms": 0, "title": 0, "prelim": 0}
    if inPlot:
        low["lumi"] = [0.73, 0.818]
        low["title"] = [0.18, 0.70]
        low["sub"] = [0.68, 0.75]
        low["cms"] = [0.182, 0.74]
        low["prelim"] = [0.27, 0.75]
        textSize["lumi"] = 0.03
        textSize["cms"] = 0.038
        textSize["title"] = 0.025
        textSize["prelim"] = 0.03
        textSize["sub"] = 0.05*0.5
    else:
        low["title"] = [0.4, 0.805]
        low["sub"] = [0.68, 0.75]
        if not pad:
            low["prelim"] = [0.24, 0.815]
            low["cms"] = [0.15, 0.807]
            low["lumi"] = [0.72, 0.818]
        else:
            low["prelim"] = [0.19, 0.815]
            low["cms"] = [0.10, 0.807]
            low["lumi"] = [0.68, 0.818]

        textSize["lumi"] = 0.03
        textSize["cms"] = 0.038
        textSize["title"] = 0.025
        textSize["prelim"] = 0.03
        textSize["sub"] = 0.05*0.5

    if adjust_sub_fit:
        low["sub"][0] += 0.1

    # Write lumi text to canvas
    lowX = low["lumi"][0]
    lowY = low["lumi"][1]
    lumi = TPaveText(lowX,lowY, lowX+0.3, lowY+0.2, "NDC")
    lumi.SetTextFont(42)
    lumi.SetTextSize(textSize["lumi"])
    lumi.SetTextColor(1)
    lumi.SetTextAlign(12)
    lumi.SetFillStyle(0)
    lumi.SetBorderSize(0)
    lumi.AddText(lumiTxt)
    lumi.DrawClone("same")

    # Write CMS text to canvas
    lowX = low["cms"][0]
    lowY = low["cms"][1]
    cmstxt = TPaveText(lowX, lowY+0.06, lowX+0.15, lowY+0.16, "NDC")
    cmstxt.SetTextFont(61)
    cmstxt.SetTextSize(textSize["cms"])
    cmstxt.SetTextColor(1)
    cmstxt.SetTextAlign(12)
    cmstxt.SetFillStyle(0)
    cmstxt.SetBorderSize(0)
    cmstxt.AddText("CMS")
    cmstxt.DrawClone("same")

    # Write title text to canvas
    lowX = low["title"][0]
    lowY = low["title"][1]
    samplestxt = TPaveText(lowX, lowY+0.06, lowX+0.3, lowY+0.16, "NDC")
    samplestxt.SetTextFont(42)
    samplestxt.SetTextSize(textSize["title"])
    samplestxt.SetTextColor(1)
    samplestxt.SetTextAlign(12)
    samplestxt.SetFillStyle(0)
    samplestxt.SetBorderSize(0)
    samplestxt.AddText(title)
    samplestxt.DrawClone("same")

    if prelim:
        # Write prelim text to canvas
        lowX = low["prelim"][0]
        lowY = low["prelim"][1]
        pretxt = TPaveText(lowX, lowY+0.05, lowX+0.15, lowY+0.15, "NDC")
        pretxt.SetTextFont(52)
        pretxt.SetTextSize(textSize["prelim"])
        pretxt.SetTextColor(1)
        pretxt.SetTextAlign(12)
        pretxt.SetFillStyle(0)
        pretxt.SetBorderSize(0)
        pretxt.AddText("Preliminary")
        pretxt.DrawClone("same")

    if sub is not None:
        # Write sub text to canvas
        lowX = low["sub"][0]
        lowY = low["sub"][1]
        subtxt = TPaveText(lowX, lowY+0.05, lowX+0.15, lowY+0.15, "NDC")
        subtxt.SetTextFont(42)
        subtxt.SetTextSize(textSize["sub"])
        subtxt.SetTextColor(1)
        subtxt.SetTextAlign(12)
        subtxt.SetFillStyle(0)
        subtxt.SetBorderSize(0)
        subtxt.AddText(sub)
        subtxt.DrawClone("same")

    # Redraw axes ticks
    gPad.RedrawAxis()

    return canvas


def plot2D(
    samples: Dict[str, pd.DataFrame],
    filename: str,
    era: str,
    path: str,
    region: str,
    ybranch: str,
    subdir: str
) -> None:
    """
    Plot 2D histogram of pT/mgg vs mgg and pT vs mgg in both EB/EE.
    """

    sorted_samples = sorted(samples.items())
    for idx, (mass, sample) in enumerate(sorted_samples):
        # Run EB/EE configuration before plotting
        if region == "combined":
            cut_p1 = [True] * len(sample)
            cut_p2 = [True] * len(sample)
        if ybranch == "dipho_mass":
            if region == "EB":
                cut_p1 = sample.lead_isScEtaEB
                cut_p2 = sample.sublead_isScEtaEB
            elif region == "EE":
                cut_p1 = sample.lead_isScEtaEE
                cut_p2 = sample.sublead_isScEtaEE
            elif region == "2EB":
                cut_p1 = sample.lead_isScEtaEB & sample.sublead_isScEtaEB
                cut_p2 = sample.lead_isScEtaEB & sample.sublead_isScEtaEB
            elif region == "2EE":
                cut_p1 = sample.lead_isScEtaEE & sample.sublead_isScEtaEE
                cut_p2 = sample.lead_isScEtaEE & sample.sublead_isScEtaEE
            elif region == "1EB1EE":
                cut_p1 = (sample.lead_isScEtaEB & sample.sublead_isScEtaEE) | (sample.lead_isScEtaEE & sample.sublead_isScEtaEB)
                cut_p2 = (sample.lead_isScEtaEB & sample.sublead_isScEtaEE) | (sample.lead_isScEtaEE & sample.sublead_isScEtaEB)

        mass_in = mass[:2]
        print(f"Mass: {mass_in} GeV")

        # Canvas setup
        canvas = TCanvas("canvas", "canvas", 1000, 1000)

        # Histogram with custom number of equal bins, bounded by (bounds[0], bounds[1])
        normalize = False

        if ybranch == "dipho_mass":
            names = [f"pT1_mgg_vs_mgg_{region}", f"pT2_mgg_vs_mgg_{region}", f"pT1_vs_mgg_{region}", f"pT2_vs_mgg_{region}"]
            branches = ["pT1_m_gg", "pT2_m_gg", "lead_pt", "sublead_pt"]
            xtitles = ["p_{T}^{#gamma_{1}} / m_{#gamma #gamma}", "p_{T}^{#gamma_{2}} / m_{#gamma #gamma}", "p_{T}^{#gamma_{1}}", "p_{T}^{#gamma_{2}}"]
            boundsList = [[0.0, 7.0, 0.0, 125.0], [0.0, 5.0, 0.0, 125.0], [0.0, 200.0, 0.0, 125.0], [0.0, 200.0, 0.0, 125.0]]
            binsScales = [[100.0/7.0, 1.0], [20.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
            cut_list = [cut_p1, cut_p2, cut_p1, cut_p2]
        elif ybranch == "mass":
            names = [f"pT1_ma1_vs_ma1_{region}", f"pT2_ma1_vs_ma1_{region}", f"pT1_ma2_vs_ma2_{region}", f"pT2_ma2_vs_ma2_{region}", f"pT1_vs_ma1_{region}", f"pT2_vs_ma1_{region}", f"pT1_vs_ma2_{region}", f"pT2_vs_ma2_{region}"]
            branches = ["pT1_ma1", "pT2_ma1", "pT1_ma2", "pT2_ma2", "LeadPs_leading_pho_pt", "LeadPs_subleading_pho_pt", "SubleadPs_leading_pho_pt", "SubleadPs_subleading_pho_pt"]
            xtitles = ["p_{T}^{#gamma_{1}} / m_{a1}", "p_{T}^{#gamma_{2}} / m_{a1}", "p_{T}^{#gamma_{1}} / m_{a2}", "p_{T}^{#gamma_{2}} / m_{a2}", "p_{T}^{#gamma_{1}}", "p_{T}^{#gamma_{2}}", "p_{T}^{#gamma_{1}}", "p_{T}^{#gamma_{2}}"]
            boundsList = [[0.0, 10.0, 10.0, 65.0], [0.0, 10.0, 10.0, 65.0], [0.0, 10.0, 10.0, 65.0], [0.0, 10.0, 10.0, 65.0], [0.0, 160.0, 10.0, 65.0], [0.0, 160.0, 10.0, 65.0], [0.0, 160.0, 10.0, 65.0], [0.0, 160.0, 10.0, 65.0]]
            binsScales = [[10.0, 5.0], [10.0, 5.0], [10.0, 5.0], [10.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]]
            cut_list = [[True] * len(sample) for _ in range(len(names))]

        assert len(names) == len(branches) == len(xtitles) == len(boundsList) == len(binsScales) == len(cut_list)
        for name, branch, xtitle, bounds, binsScale, cut in zip(names, branches, xtitles, boundsList, binsScales, cut_list):
            if ybranch == "mass":
                if "ma1" in name:
                    ybranch_loop = "LeadPs_mass"
                elif "ma2" in name:
                    ybranch_loop = "SubleadPs_mass"
            else:
                ybranch_loop = ybranch

            hist = TH2D(name, name, int((bounds[1]-bounds[0])*binsScale[0]), bounds[0], bounds[1], int((bounds[3]-bounds[2])*binsScale[1]), bounds[2], bounds[3])
            xbranch_df = sample[cut][branch].tolist()
            ybranch_df = sample[cut][ybranch_loop].tolist()

            # Fill histogram
            for x, y in zip(xbranch_df, ybranch_df):
                hist.Fill(x,y)

            # Normalize
            if normalize:
                if hist.Integral() > 0.0:
                    hist.Scale(1.0 / hist.Integral())
                else:
                    hist.Scale(1.0)
                    print("Cannot normalize empty histogram.")

            hist.SetStats(0)
            hist.SetMarkerStyle(8)
            hist.SetMarkerSize(0.5)
            hist.SetTitle(f"")
            hist.GetYaxis().SetTitleFont(42)
            if ybranch == "dipho_mass":
                hist.GetYaxis().SetTitle("m_{#gamma #gamma}" + f"  /  {1.0 / binsScale[1]: 1.3f}")
            elif ybranch_loop == "LeadPs_mass" or ybranch_loop == "SubleadPs_mass":
                if "ma1" in name:
                    hist.GetYaxis().SetTitle("m_{a1}" + f"  /  {1.0 / binsScale[1]: 1.3f}")
                elif "ma2" in name:
                    hist.GetYaxis().SetTitle("m_{a2}" + f"  /  {1.0 / binsScale[1]: 1.3f}")
            hist.GetXaxis().SetTitleFont(42)
            hist.GetXaxis().SetTitle(xtitle + f"  /  {1.0 / binsScale[0]: 1.3f}")
            hist.GetXaxis().SetTitleOffset(1.5)
            hist.Draw("COLZ")

            plotFancy(canvas, name, lumiTxt="   26.67 fb^{-1} (13.6 TeV)".strip(), prelim=True, inPlot=False, colz=True)

            #new_filename = filename.replace("_replace", f"_{name}_{mass_in}GeV_{era}")
            new_filename = filename.replace("_replace", f"{name}_{mass_in}GeV_{era}")

            if not os.path.exists(f"{path}/{subdir}/{mass_in}_GeV/"):
                os.mkdir(f"{path}/{subdir}/{mass_in}_GeV/")
            canvas.SaveAs(f"{path}/{subdir}/{mass_in}_GeV/{new_filename}")
            assert os.path.isfile(f"{path}/{subdir}/{mass_in}_GeV/{new_filename}"), "Saved plot does not exist."

            for plot in [f"pT1_mgg_vs_mgg_{region}", f"pT2_mgg_vs_mgg_{region}", f"pT1_ma1_vs_ma1_{region}", f"pT2_ma1_vs_ma1_{region}", f"pT1_ma2_vs_ma2_{region}", f"pT2_ma2_vs_ma2_{region}"]:
                if name == plot:
                    canvas.Clear()

                    # X Projection
                    hist1D = hist.ProjectionX()
                    hist1D.SetStats(0)
                    hist1D.SetMarkerStyle(8)
                    hist1D.SetMarkerSize(0.5)
                    hist1D.SetTitle(f"")
                    hist1D.GetYaxis().SetTitleFont(42)
                    hist1D.GetYaxis().SetTitle(f"Events / {1.0 / binsScale[0]: 1.3f}")
                    hist1D.GetXaxis().SetTitleFont(42)
                    hist1D.GetXaxis().SetTitle(xtitle)
                    hist1D.Draw("hist")

                    plotFancy(canvas, f"{name}", sub="projX", lumiTxt="   26.67 fb^{-1} (13.6 TeV)".strip(), prelim=True, inPlot=False, colz=True)

                    #new_filename = filename.replace("_replace", f"_{name}_{mass_in}GeV_{era}")
                    new_filename = filename.replace("_replace", f"{name}_projX_{mass_in}GeV_{era}")

                    if not os.path.exists(f"{path}/{subdir}/{mass_in}_GeV/projX/"):
                        os.mkdir(f"{path}/{subdir}/{mass_in}_GeV/projX/")
                    canvas.SaveAs(f"{path}/{subdir}/{mass_in}_GeV/projX/{new_filename}")
                    assert os.path.isfile(f"{path}/{subdir}/{mass_in}_GeV/projX/{new_filename}"), "Saved plot does not exist."
                    canvas.Clear()


                    # Legend setup
                    lsize = [0.65, 0.7, 0.9, 0.8]
                    legend = TLegend(lsize[0], lsize[1], lsize[2], lsize[3])
                    legend.SetTextFont(42)
                    legend.SetBorderSize(0)
                    legend.SetFillStyle(0)

                    # X Projection
                    hist1D = hist.ProjectionX()
                    hist1D.SetStats(0)
                    hist1D.SetTitle(f"")
                    hist1D.SetLineWidth(2)
                    hist1D.GetYaxis().SetTitleFont(42)
                    hist1D.GetYaxis().SetTitle(f"Events / {1.0 / binsScale[0]: 1.3f}")
                    hist1D.GetXaxis().SetTitleFont(42)
                    hist1D.GetXaxis().SetTitle(xtitle)
                    hist1D.SetLineColor(kRed)
                    legend.AddEntry(hist1D, f"Projection-X: {hist1D.Integral()}", "l")
                    hist1D.Draw("hist")

                    # To remove weird vertical line: apply plotFancy on one canvas then add additional histograms
                    plotFancy(canvas, f"{name}", lumiTxt="   26.67 fb^{-1} (13.6 TeV)".strip(), prelim=True, inPlot=False)

                    # Original pT/mgg plot
                    original_hist = fillHist(branch, sample[cut], bounds=bounds[:2], binsScale=binsScale[0], normalize=False)
                    original_hist.SetStats(0)
                    original_hist.SetTitle(f"")
                    original_hist.SetLineWidth(2)
                    legend.AddEntry(original_hist, f"Original Plot {original_hist.Integral()}", "l")
                    original_hist.Draw("same hist")
                    legend.Draw("same")

                    #new_filename = filename.replace("_replace", f"_{name}_{mass_in}GeV_{era}")
                    new_filename = filename.replace("_replace", f"{name}_overlay_{mass_in}GeV_{era}")

                    if not os.path.exists(f"{path}/{subdir}/{mass_in}_GeV/overlay/"):
                        os.mkdir(f"{path}/{subdir}/{mass_in}_GeV/overlay/")
                    canvas.SaveAs(f"{path}/{subdir}/{mass_in}_GeV/overlay/{new_filename}")
                    assert os.path.isfile(f"{path}/{subdir}/{mass_in}_GeV/overlay/{new_filename}"), "Saved plot does not exist."



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, type=str, help="Path of parquet file to run.")
    parser.add_argument("-d", "--dipho", required=False, action="store_true", help="Plot diphoton plots instead.")
    args = parser.parse_args()

    # Get specified directory in signal_tests
    assert os.path.exists(args.path)
    print(args.path[8:14])

    if not os.path.exists("2d_plots"):
        os.mkdir("2d_plots")

    if not args.dipho:
        ybranch = "mass"
    elif args.dipho:
        ybranch = "dipho_mass"

    to_plot = ""
    if ybranch == "mass":
        to_plot = "pseudos"
    elif ybranch == "dipho_mass":
        to_plot = "diphotons"
    if not os.path.exists(f"2d_plots/{to_plot}/"):
        os.mkdir(f"2d_plots/{to_plot}/")

    if ybranch == "dipho_mass":
        samples = {args.path[8:14]: loadSamples(args.path, branches=["dipho_mass", "pT1_m_gg", "pT2_m_gg", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_pt", "sublead_pt"])}
    elif ybranch == "mass":
        samples = {args.path[8:14]: loadSamples(args.path, branches=["LeadPs_mass", "SubleadPs_mass", "pT1_ma1", "pT2_ma1", "pT1_ma2", "pT2_ma2", "LeadPs_leading_pho_pt", "LeadPs_subleading_pho_pt", "SubleadPs_leading_pho_pt", "SubleadPs_subleading_pho_pt"])}
    #for region in ["combined", "EB", "EE", "2EB", "2EE", "1EB1EE"] if ybranch == "dipho_mass" else ["combined"]:
    for region in ["combined"]:
        plot2D(samples, f"_replace.png", 2022, f"2d_plots", region, ybranch=ybranch, subdir=to_plot)
