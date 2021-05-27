
#Addition of a new dot.
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

#Initial state. Base dataset.

##def __init__ visualizer(self):


##Falta que lo clasifique.
def add_dot(sepal_width, sepal_length, color):
    df = px.data.iris()

    newDyc = {"sepal_width": sepal_width, "sepal_length": sepal_length, "species": color}

    df = df.append(newDyc, ignore_index=True)
    newFig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    newFig.show()

   # KNN.fit(X, Y)
    #KNN.askNewData()

if __name__ == "__main__":
    df = px.data.iris()

    speciesDyc = {"setosa": 0, "versicolor": 1, "virginica": 2}

    X = df["sepal_width", "sepal_length"].values
    Y = df["species"].apply(speciesDyc.get).values

    print(np.isnan(Y).sum() == 0)

    df = df[["sepal_width", "sepal_length", "species"]]
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    fig.show()

    fig.show()