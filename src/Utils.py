def plotWeightsDistribution(data, density=False, dataType="", titlePrefix = None):

    import torch
    from matplotlib import pyplot as plt

    for label, d in data.items():
        hy, hx = torch.histogram(d, density=density)
        plt.plot(hx[:-1].detach(), hy.detach(), label=label)

    plt.title("Distribution of " + dataType + (" - " + titlePrefix if titlePrefix else ""))
    plt.xlabel("Weight Value")
    if density:
        plt.ylabel("Density")
    else:
        plt.ylabel("Frequency")
    plt.legend()

    return plt


def plotActivationDistribution(activation, density=False, dataType="", titlePrefix = None):

    plotData = {}
    for label, act_list in activation.items():
        plotData[label] = act_list

    plt = plotWeightsDistribution(data=plotData, density=density, dataType=dataType, titlePrefix=titlePrefix)

    return plt
