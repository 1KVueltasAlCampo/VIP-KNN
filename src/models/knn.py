from base_models import SupervisedModel

import numpy as np
import panda as pd


class KNN(SupervisedModel):
    """
               Creates a new KNN object.

               Arguments:
                 K: the number of nearest neighboors
    """


    def __int__(self, k):
        self.k = k

    def get_k_neighboors(self, datapoint):
        """
       Gets the k-nearest neighboors of a given datapoint
       Argunments:
         datapoint: numpy.array, a row vector
       Returns:
         indices: list, indices corresponding with the k datapoints in self.X most
                  similar to datapoint
    """
        distances = []  # distances between the matrix and the datapoint

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

    """
       Calculates the euclidean 
       Arguments:
         datapoint1: numpy.array, first datapoint. It's the row vector we want to compare with the others.
         datapoint2: numpy.array, second datapoint
       Returns:
         Distance between the given datapoints
       """

    def calculate_distance(self, datapoint1, datapoint2):
        if isinstance(datapoint1, np.ndarray) and isinstance(datapoint2, np.ndarray):
            array3 = np.subtract(datapoint2, datapoint1)
            return np.linalg.norm(array3)
        else:
            raise ValueError(" Datatype not valid")

    def step_fit(self,main_matrix, y):
        self.data = main_matrix
        self.classes = y

    """
     Predicts the class for each datapoint in the matrix X.
        Arguments:
        X: numpy.ndarray, matrix used to get predictions for each datapoint, where each row represents a datapoint.
     Returns:
       predictions: numpy.ndarray, class predicted for each datapoint in X
    """

    def step_predict(self, x):
        preds = []
        for datapoint in x:
            indices = self.get_k_nearest_neighboors(datapoint)
            # Obtener los indices de las clases
            classes = np.array([self.classes[idX] for idX in indices])
            # Obtener la clase mas frecuente de los vecinos mas cercanos
            counts = np.bincount(classes)
            predicted_class = np.argmax(counts)
            preds.append(predicted_class)
        return np.array(preds)
