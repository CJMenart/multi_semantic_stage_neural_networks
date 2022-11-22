# Going from csv dataframe to plots for slides/papers/etc
import seaborn
import pandas
from matplotlib import pyplot as plt
import textwrap 

def plot_initial_results(csv_path,y='Accuracy'):
    """
    Basic NGM and benchmark ResNet at three different sizes each
    """
    param_numbers = [11_184_720,  21_292_880, 29_227_385,  48_780_921, 66_867_024, 148_833_494]
    param_number_labels = ['11M', '21M', '29M', '48M', '66M', '148M']
    df = pandas.read_csv(csv_path)
    df = df.loc[df["Model Type"] == "NGM"].append(df.loc[df["Model Type"] == "ResNet"])
    print(f"df: {df}")
    seaborn.set_theme()
    #plot = seaborn.relplot(x="Num. Parameters", y="Accuracy", hue="Model Type", kind="line", markers=['x','o'], data=df, legend=False)
    plot = seaborn.lineplot(x="Num. Parameters", y=y, hue="Model Type", markers=True, data=df, style='Model Type', legend=False)
    plt.legend(['NGM', 'ResNet'])
    plt.title("Neural Graphical Model versus ResNet")
    plot.set(xlabel='Num. Parameters')
    plt.xticks(param_numbers)
    if y=='Accuracy':
        plt.ylim(0.5,1)
    else:
        plt.ylim(0.0,1)
    plot.set_xticklabels(param_number_labels, rotation=45)
    plt.xlabel("Num. Parameters")
    plt.tight_layout()
    plot.get_figure().savefig("ngm_initial_results.png")
    
    
def plot_partially_sup_results(csv_path,y='Accuracy'):
    """
    Bar chart of performance of partially supervised models
    """
    df = pandas.read_csv(csv_path)
    seaborn.set_theme()
    plot = seaborn.barplot(x="Model Type",y=y,data=df)
    # plt.legend?
    plt.title("Partially Supervised NGM")
    #plt.ylim(0.3,0.9)
    if y=='Accuracy':
        plt.ylim(0.5,1)
    else:
        plt.ylim(0.0,1)
    MAX_WIDTH = 12
    plot.set_xticklabels(textwrap.fill(x.get_text(), MAX_WIDTH) for x in plot.get_xticklabels())
    plt.tight_layout()
    plot.get_figure().savefig("ngm_partially_sup.png")


def plot_supervision_chance_results(csv_path,y='Accuracy'):
    """
    Basic NGM and benchmark ResNet at three different sizes each
    """
    chance_numbers = [0.2,  0.35, 0.5, 0.65, 0.8]
    #chance_labels = ['11M', '21M', '29M', '48M', '66M', '148M']
    df = pandas.read_csv(csv_path)
    #df = df.loc[df["Model Type"] == "NGM"].append(df.loc[df["Model Type"] == "ResNet"])
    seaborn.set_theme()
    plot = seaborn.lineplot(x="Supervision Chance", y=y, hue="Model Type", markers=True, data=df, style='Model Type', legend=False)
    #plt.legend(['NGM', 'ResNet'])
    plt.legend(['Backprop REINFORCE w/Step Schedule','EM w/Linear Schedule','Partially Supervised MTL'])
    plt.title("NGM and MTL with Varying Amounts of Supervision")
    plt.xticks(chance_numbers)
    if y=='Accuracy':
        plt.ylim(0.5,1)
    else:
        plt.ylim(0.0,1)
    plt.tight_layout()
    plot.get_figure().savefig("ngm_supervision_chance_results.png")
