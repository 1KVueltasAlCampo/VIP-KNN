
#Addition of a new dot.
import plotly.express as px
import plotly.graph_objects as go

#Initial state. Base dataset.
df = px.data.iris()

df = df[["sepal_width", "sepal_length", "species"]]
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()

fig.show()

##Método que reciba un punto y lo añada.

def add_dot(sepal_width, sepal_length, color):
    df = px.data.iris()

    newDyc = {"sepal_width": sepal_width, "sepal_length": sepal_length, "species": color}

    df = df.append(newDyc, ignore_index=True)
    newFig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    newFig.show()

