
import torch
from torch import nn
import pytorch_lightning as pl




class SummationMPNN(pl.LightningModule):


    def __init__(self, node_features, edge_features, message_size, message_passes, out_features):
        super(SummationMPNN, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.message_size = message_size
        self.message_passes = message_passes
        self.out_features = out_features


    def message_terms(self, nodes, node_neighbours, edges):
        raise NotImplementedError


    def update(self, nodes, messages):
        raise NotImplementedError


    def readout(self, hidden_nodes, input_nodes, node_mask):
        raise NotImplementedError

    def forward(self, adjacency, nodes, edges):
        edge_batch_batch_indices, edge_batch_node_indices, edge_batch_neighbour_indices = adjacency.nonzero().unbind(-1)
        node_batch_batch_indices, node_batch_node_indices = adjacency.sum(-1).nonzero().unbind(-1)

        same_batch = node_batch_batch_indices.view(-1, 1) == edge_batch_batch_indices
        same_node = node_batch_node_indices.view(-1, 1) == edge_batch_node_indices
        message_summation_matrix = (same_batch * same_node).float()

        edge_batch_edges = edges[edge_batch_batch_indices, edge_batch_node_indices, edge_batch_neighbour_indices, :]
        hidden_nodes = nodes.clone()
        node_batch_nodes = hidden_nodes[node_batch_batch_indices, node_batch_node_indices, :]

        for i in range(self.message_passes):
            edge_batch_nodes = hidden_nodes[edge_batch_batch_indices, edge_batch_node_indices, :]
            edge_batch_neighbours = hidden_nodes[edge_batch_batch_indices, edge_batch_neighbour_indices, :]

            message_terms = self.message_terms(edge_batch_nodes, edge_batch_neighbours, edge_batch_edges)
            messages = torch.matmul(message_summation_matrix, message_terms)
            node_batch_nodes = self.update(node_batch_nodes, messages)

            hidden_nodes[node_batch_batch_indices, node_batch_node_indices, :] = node_batch_nodes

        node_mask = (adjacency.sum(-1) != 0)
        output = self.readout(hidden_nodes, nodes, node_mask)
        return output
