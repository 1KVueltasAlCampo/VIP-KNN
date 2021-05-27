from base_models import SupervisedModel
import

import numpy as np


class KNN(SupervisedModel):

    def __init__(self, k):
        """
            Creates a new KNN object.
           Arguments:
                k: the number of nearest neighboors

        """
        super(KNN, self).__init__()
        self.k = k
        self.classes = None
        self.data = None

        KNN.askNewData(self)

    def __str(self):
        return f"[KNN Object [k={self.k}]"

    def calculate_distance(self, datapoint1, datapoint2):
        """
               Calculates the euclidean
               Arguments:
                 datapoint1: numpy.array, first datapoint. It's the row vector we want to compare with the others.
                 datapoint2: numpy.array, second datapoint
               Returns:
                 Distance between the given datapoints
            """
        if isinstance(datapoint1, np.ndarray) and isinstance(datapoint2, np.ndarray):
            array3 = np.subtract(datapoint2, datapoint1)
            return np.linalg.norm(array3)
        else:
            raise ValueError(" Datatype not valid")

    def get_k_neighbours(self, datapoint):
        """
       Gets the k-nearest neighboors of a given datapoint
       Argunments:
         datapoint: numpy.array, a row vector
       Returns:
         indices: list, indices corresponding with the k datapoints in self.X most
                  similar to datapoint
        """
        distances = []  # distances between the matrix and the datapoint
        print(self.data)
        size = len(self.data)
        vector_of_the_matrix = []

        for i in range(size):
            vector_of_the_matrix = self.data[i]
            np_vector_of_the_matrix = np.array(vector_of_the_matrix)
            two_vectors_difference = self.calculate_distance(datapoint, np_vector_of_the_matrix)
            distances.append(two_vectors_difference)

        distances = np.array(distances)
        indices = distances.argsort()
        k_indices = []
        for i in range(self.k):
            k_indices.append(indices[i])

        return k_indices


    def step_fit(self, main_matrix, y):
       pass

    def fit(self, main_matrix, classes):
        self.data = main_matrix
        self.classes = classes

    def predict(self, x):
        preds = []
        for datapoint in x:
            pred = self.step_predict(datapoint)
            preds.append(pred)
        return np.array(preds)

    # Fin de los abstractos.
    def step_predict(self, datapoint):
        indices = self.get_k_neighbours(datapoint)
        # Obtener los indices de las clases
        classes = np.array([self.classes[idX] for idX in indices])
        # Obtener la clase mas frecuente de los vecinos mas cercanos
        counts = np.bincount(classes)
        predicted_class = np.argmax(counts)
        return predicted_class

    def askNewData(self):
        print("Please, type the x value")
        x = input()
        print("Please, type the y value")
        y = input()

        inputs = [x, y]
        array_inputs = np.array(inputs)

        KNN.predict(self, array_inputs)

if __name__ == "__main__":
    object_knn = KNN(4)
    print(object_knn)
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    object_knn.fit(X_train, y_train)
    preds = object_knn.predict(X_test)
    acc = (preds == y_test).sum() / len(preds)
    print(preds)
    print(f"Accuracy: {acc}")