import pickle
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot
from prettytable import PrettyTable


def compMet(files : List[str], outputDir = "complog"):
    
    simMetrics = {}
    for file in files:
        simName = Path(file).name
        with open(file, "rb") as simpk:
            simMetrics[simName] = pickle.load(simpk)

    def helperDrawer(metDict, keys, maxY, autoYlim, ylim):
        fig = []
        for key in keys:
            fig.append(pyplot.figure(key))
            pyplot.title(key)
            x, y = metDict["frame"], metDict[key]
            pyplot.plot(x,y)

            if autoYlim:
                if max(y) > maxY:
                    maxY = max(y)
                ylim = [-maxY/10, maxY+5]
            pyplot.ylim(ylim)
        return maxY, fig

    def helperSaver(figs, groups):
        for fig, name in zip(figs, groups):
            lgd = fig.gca().legend(sims, loc="best")
            fig.savefig(
                Path(outputDir).joinpath(name).as_posix(),
                format="png",
                bbox_extra_artists=(lgd)
            )


    def floatfmt(num, places = 4):
        return f"{num:.{places}f}"

    maxY = 0
    sims = simMetrics.keys()
    figGrp1 = ["gtFrame", "sucBB", "sucDet", "sucAtt", "missBB"]
    figGrp2 = ["TPR", "AA", "ACAC", "ACTC", "NTE"]
    compTable = PrettyTable(
       
        field_names= ["Sim#"] + figGrp2
    )
    for simName in sims:
        maxY, figs1 = helperDrawer(
            simMetrics[simName], 
            figGrp1,
            maxY, True, []
        )

        _, figs2 = helperDrawer(
            simMetrics[simName],
            figGrp2,
            0, False, [-0.1, 1.1]
        )

        compTable.add_row(
            [
                simName,
                floatfmt(simMetrics[simName]["TPR"][-1]),
                floatfmt(simMetrics[simName]["AA"][-1]),
                floatfmt(np.nan_to_num(np.divide(np.sum(simMetrics[simName]["ACAC"]), np.count_nonzero(simMetrics[simName]["ACAC"])))),
                floatfmt(np.nan_to_num(np.divide(np.sum(simMetrics[simName]["ACTC"]), np.count_nonzero(simMetrics[simName]["ACTC"])))),
                floatfmt(np.nan_to_num(np.divide(np.sum(simMetrics[simName]["NTE"]), np.count_nonzero(simMetrics[simName]["NTE"])))),
            ]
        )

    Path(outputDir).mkdir(parents=True, exist_ok=True)
    helperSaver(figs1, figGrp1)
    helperSaver(figs2, figGrp2)    
    pyplot.show(block=False)
    print(compTable)
    
    with open(
        Path(outputDir).joinpath("comptbl.html").as_posix(), 
        "w+"
    ) as file:
        file.write(
            compTable.get_html_string(attributes={"class":"table"}, format=True)
        )