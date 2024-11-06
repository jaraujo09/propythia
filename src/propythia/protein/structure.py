import torch
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
from graphein.protein.config import ProteinGraphConfig
import graphein.protein as gp
from torch.utils.data import Dataset
from torch_geometric.utils.convert import from_networkx
from typing import List, Dict, Any
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from graphein.protein.utils import download_alphafold_structure
from Bio.PDB import PDBList
from networkx.classes.graph import Graph as Graph_NetworkX


class Graph:
    def __init__(self, identifier: str, id_type: str = 'pdb', cluster_info: Dict[str, Any] = None):
        self.identifier = identifier
        self.id_type = id_type
        self.pdb_file = None
        self.graph = None
        self.data = None
        self.cluster_info = cluster_info
        self.config = self.default_config()
        self.download_structure()
        self.construct_my_graph()

    def default_config(self) -> ProteinGraphConfig:
        return ProteinGraphConfig(
            edge_construction_functions=[
                gp.add_peptide_bonds,
                gp.add_aromatic_interactions,
                gp.add_hydrogen_bond_interactions,
                gp.add_disulfide_interactions,
                gp.add_ionic_interactions,
                gp.add_aromatic_sulphur_interactions,
                gp.add_cation_pi_interactions,
                gp.add_hydrophobic_interactions,
                gp.add_vdw_interactions,
                gp.add_backbone_carbonyl_carbonyl_interactions,
                gp.add_salt_bridges,
                Graph.add_distance_threshold_with_custom_threshold
            ],
            node_metadata_functions=[gp.amino_acid_one_hot]
        )
    
    @staticmethod
    def add_distance_threshold_with_custom_threshold(graph, threshold=4.0, long_interaction_threshold=5.0):
        return gp.add_distance_threshold(graph, threshold=threshold, long_interaction_threshold=long_interaction_threshold)


    def download_structure(self):
        try:
            if self.id_type == 'pdb':
                pdbl = PDBList()
                self.pdb_file = pdbl.retrieve_pdb_file(self.identifier, file_format='pdb', pdir='pdb_files/')
                print(f"PDB file downloaded: {self.pdb_file}")
            elif self.id_type == 'uniprot':
                result = download_alphafold_structure(self.identifier, out_dir="pdb_files/")
                if isinstance(result, tuple):
                    self.pdb_file = result[0]
                else:
                    self.pdb_file = result
                print(f"AlphaFold file downloaded: {self.pdb_file}")
        except Exception as e:
            print(f"Error downloading structure: {e}")
            raise

    def construct_my_graph(self):
        try:
            self.graph = gp.construct_graph(path=self.pdb_file, config=self.config)
            if self.graph.name is None:
                self.graph.name = self.identifier
            self.data = self.convert_to_pyg_data()
        except Exception as e:
            print(f"Error constructing graph: {e}")
            raise

    def convert_to_pyg_data(self):
        try:
            node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}
            nodes = list(self.graph.nodes(data=True))
            edges = list(self.graph.edges(data=True))

            x = torch.tensor([node[1]['coords'] for node in nodes], dtype=torch.float)
            edge_index = torch.tensor([[node_to_idx[edge[0]], node_to_idx[edge[1]]] for edge in edges], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([edge[2]['distance'] for edge in edges], dtype=torch.float).view(-1, 1)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            return data
        except Exception as e:
            print(f"Error converting to PyG data: {e}")
            raise

    def get_dataloader(self, batch_size: int = 1, shuffle: bool = False):
        return PyGDataLoader([self.data], batch_size=batch_size, shuffle=shuffle)

    @classmethod
    def batch_graphs(cls, identifiers: List[str], id_type: str = 'pdb', config: Dict[str, Any] = None):
        graphs = []
        for id in identifiers:
            try:
                graph = cls(identifier=id, id_type=id_type, config=config)
                graphs.append(graph.data)
            except Exception as e:
                print(f"Skipping ID {id} due to error: {e}")
        return PyGDataLoader(graphs, batch_size=len(graphs), shuffle=False)

class ProteinGraphDataset(Dataset, list[Graph]):
    def __init__(self, df:Dataset, graph_list:list[Graph]=None):
        try:
            if graph_list is not None: 
                if len(graph_list) == 0 or len(graph_list) != len(df) or all(isinstance(graph, Graph_NetworkX) for graph in graph_list) != True :
                    raise ValueError(f"graph_list must be a list of Graph objects with the same length as the DataFrame, val1:{len(graph_list) == 0}"+ 
                                    f"val2:{len(graph_list)}, val3:{all(isinstance(graph, Graph) for graph in graph_list)}")
            
            self.graph_list = graph_list
            self.df = df
            self.label_dict = pd.Series(df.label.values, index=df.id).to_dict()        
            self.graphs = []
            self._load_all_graphs()
            self.y = df.label.values
            
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
            raise
    
    def _load_all_graphs(self):
        # Instanciate a Graph object for each row in the dataframe
        graphs = []
        if self.graph_list is not None:
            graphs = self.graph_list
        else:
            for idx, row in self.df.iterrows():
                try:
                    graph = Graph(identifier=row['id'], id_type='uniprot')
                    print(graph)
                    if graph.graph.graph['name'] in self.label_dict:
                        graphs.append(graph.graph)
                except Exception as e:
                    print(f"Error loading graph: {e}")
                    continue
        self.graphs.extend(graphs)

    def __len__(self):
        return len(self.graphs)

    def __iter__(self):
            for index in range(self.__len__()):
                yield self.__getitem__(index)
    def toList(self)-> list[Data]:
            out = []
            for index in range(self.__len__()):
                out.append(self.__getitem__(index))
            return out

    def __getitem__(self, idx):
        try:
            graph = self.graphs[idx]

            nodes = list(graph.nodes(data=True))

            graph.y = torch.tensor([self.y[idx]], dtype=torch.long)

            x = torch.tensor([node[1]['coords'] for node in nodes], dtype=torch.float)

            pyg_data = self._to_pyg_data(graph)

            pyg_data.x = x

            id = graph.graph['name']

            pyg_data.y = torch.tensor([self.label_dict[id]], dtype=torch.long)  # Adiciona o rótulo
            return pyg_data
        except Exception as e:
            print(f"Error converting to PyG data: {e}")
            raise
        
    def _to_pyg_data(self, graph)->Data:
        if isinstance(graph, nx.Graph):
            data = from_networkx(graph)
            return data
        else:
            raise ValueError("Graph is not a NetworkX graph")
