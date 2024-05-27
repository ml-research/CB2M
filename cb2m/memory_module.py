import torch
import numpy as np
import scipy
import pandas as pd
import faiss


class MemoryModule:

    def __init__(self, k):
        self.knowledge_base = pd.DataFrame(columns=['key', 'intervention'])
        self.data = []
        self.key_size = 0
        self.k = k

        self.gpu_resource = faiss.StandardGpuResources()  # use a single GPU
        self.gpu_index = None

    def add_intervention(self, key, intervention):
        """
        Adds a new entry to the memory. The key is the encoding of the example and intervention is the intervention
        applied to the input. The intervention is a tuple of the form (concepts, concept values).
        """
        dict_key = key
        if type(dict_key) == torch.Tensor:
            dict_key = dict_key.detach().cpu().numpy()
            dict_key = dict_key.flatten()

        self.data.append({'key': dict_key, 'intervention': intervention})

    def convert_data(self):
        """
        Converts all stored information in self.data (e.g. by the add_concept method) to a dataframe.
        Afterward, self.data is reset.
        """
        self.knowledge_base = pd.concat([self.knowledge_base,
                                         pd.DataFrame(self.data, columns=['key', 'intervention'])])

        self.key_size = self.data[0]['key'].shape[0]
        self.data = []

    def prepare_eval(self):
        """
        Prepares the retriever for retrieval, i.e. wraps the data conversion and setup for the knn.
        use_prototype describes if prototypes are used. num_clusters is the number of additional cluster
        prototypes, if 'use_prototype' is True.
        """
        self.convert_data()
        # build the faiss index:

        cpu_index = faiss.IndexFlatL2(self.key_size)  # create a CPU index
        self.gpu_index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, cpu_index)  # transfer the index to GPU

        self.gpu_index.add(np.stack(self.knowledge_base['key'].tolist()))  # add vectors to the index

    def check_intervention(self, key, threshold):
        """
        Check if the key should get an interventions (is probably misclassified).
        This happens if there are at least k neighbors within the distance threshold of the key in the knowledge base
        (if the k closest neighbor is within the distance threshold).
        If so, return true, otherwise False.
        """
        return threshold >= self.closest_distance(key)

    def closest_distance(self, key):
        """
        For a given key, returns the distance to the k closest neighbor. (E.g. distance to the third-closest
        neighbor if k equals 3).
        """
        if type(key) == torch.Tensor:
            key = key.detach().cpu().numpy()
        key = key.reshape(1, -1)

        distances, _ = self.gpu_index.search(key, self.k)
        distances = np.sqrt(distances[0])
        return distances[self.k - 1]

    def get_intervention(self, key, threshold):
        """
        Check if there is a stored interventions which was applied to an encoding with distance < threshold.
        If there is one, return the intervention, else return None
        """
        if type(key) == torch.Tensor:
            key = key.detach().cpu().numpy()
        key = key.reshape(1, -1)

        # retrieve the nn
        distances, neighbors = self.gpu_index.search(key, self.k)
        nns = [self.knowledge_base.iloc[neighbors[0][i]] for i in range(self.k)]
        distances = np.sqrt(distances[0])

        candidates = [nns[i] for i in range(self.k) if distances[i] < threshold]
        if len(candidates) >= self.k:
            distances_nn = [d for d in distances if d < threshold]
            interventions_nn = [c['intervention'] for c in candidates]
            intervention_idx = self.single_combination(distances_nn, interventions_nn)
        else:
            intervention_idx = None
            
        return intervention_idx

    def reset_kb(self):
        self.knowledge_base = pd.DataFrame(columns=['key', 'concept_order'])

    @staticmethod
    def single_combination(distances, interventions_nn):
        """
        Return the intervention of the closest neighbor
        """
        closest_idx = distances.index(min(distances))
        return interventions_nn[closest_idx]
