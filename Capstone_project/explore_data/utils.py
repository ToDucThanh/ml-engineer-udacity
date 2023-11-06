import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

from ipywidgets import interact
from typing import List
from IPython.display import display

FONT_SIZE_TICKS = 15
FONT_SIZE_TITLE = 15
FONT_SIZE_AXES = 15


def plot_histogram(df: pd.DataFrame, features: List[str], bins: int = 16):
    """Create interactive histogram plots for the dataset.

    Args:
        df (pd.DataFrame): The dataset.
        features (List[str]): List of features to include in the plot.
        bins (int, optional): Number of bins in the histograms. Defaults to 16.
    """

    def _plot(feature):
        plt.figure(figsize=(18, 6))
        x = df[feature].values
        plt.xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        sns.histplot(x, bins=bins)
        plt.ylabel(f"Count", fontsize=FONT_SIZE_AXES)
        plt.title(f"Histogram of feature {feature}", fontsize=FONT_SIZE_TITLE)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    interact(_plot, feature=feature_selection)


def plot_box_violin(df: pd.DataFrame, features: List[str]):
    """Create interactive violin/box plots for the dataset.

    Args:
        df (pd.DataFrame): The dataset.
        features (List[str]): List of features to include in the plot.
    """

    def _plot(feature="Age", plot_type="Box"):
        plt.figure(figsize=(18, 6))
        scale = "linear"
        plt.yscale(scale)
        
        if plot_type == "Violin":
            sns.violinplot(data=df, y=feature, color="seagreen")
            
        elif plot_type == "Box":
            sns.boxplot(data=df, y=feature, color="seagreen")
            
        plt.title(f"{plot_type} plot of feature {feature}", fontsize=FONT_SIZE_TITLE)
        plt.ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)

        plt.show()

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    plot_type_selection = widgets.Dropdown(
        options=["Violin", "Box"], 
        description="Plot Type"
    )

    interact(_plot, feature=feature_selection, plot_type=plot_type_selection)


def scatterplot(df: pd.DataFrame, features: List[str]):
    """Create interactive scatterplots of the data.

    Args:
        df (pd.DataFrame): The dataset.
        features (List[str]): List of features to include in the plot.
    """
    
    def _plot(var_x, var_y):
        plt.figure(figsize=(18, 6))
        x = df[var_x].values
        y = df[var_y].values

        plt.plot(
            x, y,
            marker='o', 
            markersize=3, 
            markerfacecolor='blue', 
            markeredgewidth=1,
            linestyle='', 
            alpha=0.5
        )
        
        plt.xlabel(var_x, fontsize=FONT_SIZE_AXES)
        plt.ylabel(var_y, fontsize=FONT_SIZE_AXES)

        plt.title(f"Scatterplot of {var_x} against {var_y}", fontsize=FONT_SIZE_TITLE)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    x_selection = widgets.Dropdown(
        options=features, 
        description="X-Axis"
    )

    y_selection = widgets.Dropdown(
        options=features, 
        description="Y-Axis"
    )

    interact(_plot, var_x=x_selection, var_y=y_selection)


def correlation_matrix(df: pd.DataFrame):
    """Create correlation matrix for the dataset.

    Args:
        df (pd.DataFrame): The dataset.
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True, cbar=False, cmap="RdBu", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.show()

    
def plot_pairplot(df: pd.DataFrame, features: List[str]):
    """Create a pairplot of the features.

    Args:
        df (pd.DataFrame): The dataset
        features (List[str]): List of features to include in the plot.
    """
    with sns.plotting_context(rc={"axes.labelsize":20}):
        sns.pairplot(df[features])
    plt.show()


def plot_pie_charts(df: pd.DataFrame, 
                   features: List[str], 
                   rows : int,
                   cols : int,
                   specs: List):
    """Create pie charts of the features.

    Args:
        df (pd.DataFrame): The dataset.
        features (str): List of features to include in the plot.
        rows (int): Number of rows used to make subplots.
        cols (int): Number of cols used to make subplots.
        specs (List): Specifications used in plotly.subplots.make_subplots
    """    
    # Create subplots
    fig = make_subplots(rows=rows, cols=cols, specs=specs)
    i = 1
    j = 0
    for feature in features:
        if j < cols:
            j += 1
        else:
            j = 1
            if i < rows:
                i += 1
            else:
                raise Exception("The number of features exceeds the number of subplots, \
                                please increase rows and columns")
                
        fig.add_trace(
            (go.Pie(labels=df[feature].value_counts().keys(), 
                    values=df[feature].value_counts().values,
                    name=feature,
                    textfont = dict(size = 10),
                    hole = .4,
                    textinfo="label+percent",
                    hoverinfo="label+percent+name"
                   )
            ), 
            row = i, col = j
        )

    fig.update_layout(
        plot_bgcolor="#03fcc6",
        autosize=False,
        width=500,
        height=800,
        margin=dict(t=0, b=20, l=20, r=20))
    
    fig.show()
    

def generate_heart_disease_scatterplots(df: pd.DataFrame, features: List[str]):
    """

    Args:
        df (pd.DataFrame): The dataset
        features (List[str]): List of features to include in the plot.
    """
    sns.set_theme(
        rc={"figure.dpi": 120, 
            "axes.labelsize": 8,  
            "grid.color": "#e7fffd",
           },
        font_scale=0.7
    )
    
    # Create subplots
    fig, ax = plt.subplots(len(features), 1, figsize=(8, 16))

    for idx, (column, axes) in enumerate(zip(features, ax.flatten())):
        # Create a scatter plot
        sns.scatterplot(
            ax=axes,
            y=df.index,
            x=df[column],
            hue=df["HeartDisease"],
            palette="viridis",
            alpha=0.7
        )

    for axes in ax.flatten()[len(features):]:
        axes.set_visible(False)

    plt.tight_layout()
    plt.show()

    
def generate_heart_disease_countplot(df: pd.DataFrame, features: List[str]):
    sns.set_theme(
        rc={"figure.dpi": 120, 
            "axes.labelsize": 8,  
            "grid.color": "#e7fffd", 
           }, 
        font_scale = 0.7)
    
    fig, ax = plt.subplots(len(features), 1, figsize=(8, 16))

    for indx, (column, axes) in enumerate(zip(features, ax.flatten())):
        sns.countplot(
            ax=axes, 
            x=df[column], 
            hue=df['HeartDisease'], 
            palette="viridis",
            alpha=0.7
        )

    for axes in ax.flatten()[len(features):]:
        axes.set_visible(False)
    
    plt.tight_layout()
    plt.show()

    
def generate_heart_disease_histogram(df: pd.DataFrame, features: List[str]):
    """

    Args:
        df (pd.DataFrame): The dataset
        features (List[str]): List of features to include in the plot.
    """    
    sns.set_theme(
        rc={"figure.dpi": 120, 
            "axes.labelsize": 8,  
            "grid.color": "#e7fffd", 
           }, 
        font_scale = 0.7)

    # Create subplots
    fig, ax = plt.subplots(len(features), 1, figsize=(8, 16))

    for indx, (column, axes) in enumerate(zip(features, ax.flatten())):
        # Create a histogram plot
        sns.histplot(data=df, 
                     x=column, 
                     hue="HeartDisease", 
                     palette="viridis", 
                     alpha=0.7, 
                     multiple="stack", 
                     ax=axes)

        legend = axes.get_legend()
        handles = legend.legendHandles
        legend.remove()
        axes.legend(handles, ['0', '1'], title="HeartDisease", loc="upper right")

        # Calculate quantiles and add vertical lines
        quantiles = np.quantile(df[column], [0, 0.25, 0.50, 0.75, 1])
        for q in quantiles:
            axes.axvline(x=q, linewidth=0.5, color='r')

    plt.tight_layout()
    plt.show()
    
    
def group_by_gender(df):
    df = df.groupby("Sex").agg(
        {"Age": "mean", 
         "ChestPainType": "count",
         "RestingBP": "mean",
         "Cholesterol": "mean",
         "FastingBS": "sum",
         "RestingECG": "count",
         "MaxHR": "mean",
         "ExerciseAngina": "count",
         "Oldpeak": "mean",
         "ST_Slope": "count",
         "HeartDisease": "sum"
        }
    )
    fig = px.bar(data_frame=df, barmode="group", template="seaborn")
    fig.show()


    