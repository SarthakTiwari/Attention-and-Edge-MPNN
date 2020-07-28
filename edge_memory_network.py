import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from preprocessing import smile_to_graph,molgraph_collate_fn
from modules import GraphGather, FeedForwardNetwork
from emn_aggregation import EMN
from torch.utils.data import DataLoader
criterion = nn.MSELoss()
from sklearn.metrics import r2_score
metric = r2_score
import GPyOpt

# commented lines for classification tasks

# criterion = nn.BCELoss()
# from pytorch_lightning.metrics.classification import AUROC
# from tensorflow.keras.metrics import AUC
# m = nn.Sigmoid()
# from pytorch_lightning.metrics.functional import accuracy
# metric = AUC(multi_label=True)

torch.manual_seed(1234)
np.random.seed(1234)


class EMNImplementation(EMN):

    def __init__(self, node_features, edge_features, message_passes, out_features,
                 edge_embedding_size,
                 edge_emb_depth=3, edge_emb_hidden_dim=150, edge_emb_dropout_p=0.0,
                 att_depth=3, att_hidden_dim=80, att_dropout_p=0.0,
                 msg_depth=3, msg_hidden_dim=80, msg_dropout_p=0.0,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=100, gather_att_dropout_p=0.0,
                 gather_emb_depth=3, gather_emb_hidden_dim=100, gather_emb_dropout_p=0.0,
                 out_depth=2, out_hidden_dim=100, out_dropout_p=0, out_layer_shrinkage=1.0):
        super(EMNImplementation, self).__init__(
            edge_features, edge_embedding_size, message_passes, out_features
        )
        self.embedding_nn = FeedForwardNetwork(
            node_features * 2 + edge_features, [edge_emb_hidden_dim] * edge_emb_depth, edge_embedding_size, dropout_p=edge_emb_dropout_p
        )

        self.emb_msg_nn = FeedForwardNetwork(
            edge_embedding_size, [msg_hidden_dim] * msg_depth, edge_embedding_size, dropout_p=msg_dropout_p
        )
        self.att_msg_nn = FeedForwardNetwork(
            edge_embedding_size, [att_hidden_dim] * att_depth, edge_embedding_size, dropout_p=att_dropout_p
        )


        self.gru = nn.GRUCell(edge_embedding_size, edge_embedding_size, bias=False)
        self.gather = GraphGather(
            edge_embedding_size, gather_width,
            gather_att_depth, gather_att_hidden_dim, gather_att_dropout_p,
            gather_emb_depth, gather_emb_hidden_dim, gather_emb_dropout_p
        )
        out_layer_sizes = [ 
            round(out_hidden_dim * (out_layer_shrinkage ** (i / (out_depth - 1 + 1e-9)))) for i in range(out_depth)
        ]
        self.out_nn = FeedForwardNetwork(gather_width, out_layer_sizes, out_features, dropout_p=out_dropout_p)

    def preprocess_edges(self, nodes, node_neighbours, edges):
        cat = torch.cat([nodes, node_neighbours, edges], dim=1)
        return torch.tanh(self.embedding_nn(cat))

    def propagate_edges(self, edges, ingoing_edge_memories, ingoing_edges_mask):
        BIG_NEGATIVE = -1e6
        energy_mask = ((1 - ingoing_edges_mask).float() * BIG_NEGATIVE).unsqueeze(-1)

        cat = torch.cat([edges.unsqueeze(1), ingoing_edge_memories], dim=1)
        embeddings = self.emb_msg_nn(cat)

        edge_energy = self.att_msg_nn(edges)
        ing_memory_energies = self.att_msg_nn(ingoing_edge_memories) + energy_mask
        energies = torch.cat([edge_energy.unsqueeze(1), ing_memory_energies], dim=1)
        attention = torch.softmax(energies, dim=1)

        message = (attention * embeddings).sum(dim=1)
        return self.gru(message) 

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        return self.out_nn(graph_embeddings)


    def prepare_data(self):
        data = pd.read_csv('t_half.csv' )
        data_list=[]
        
        for index in range(len(data)):
           adjacency, nodes, edges = smile_to_graph(data['Smiles'][index])
           #targets = data.iloc[index,1:]
           targets=np.expand_dims(data['Standard Value'][index], axis=0)
           data_list.append(((adjacency, nodes, edges), targets))

        l=len(data)
        shuff=list(range(l))
        random.shuffle(shuff)
        data_list1=[data_list[i] for i in shuff ]
        self.emn_train=tuple(data_list1[:int(l*0.8)])
        self.emn_val=tuple(data_list1[int(l*0.8):int(l*0.9)])
        self.emn_test=tuple(data_list1[int(l*0.9):])


    def train_dataloader(self):
        return DataLoader(self.emn_train, batch_size=50, shuffle=True, collate_fn=molgraph_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.emn_val, batch_size=20, shuffle=False, collate_fn=molgraph_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.emn_test, batch_size=20, shuffle=False, collate_fn=molgraph_collate_fn)
      
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
      
    def training_step(self,batch,batch_idx):
        adjacency, nodes, edges, target = batch
        output = self.forward(nodes, edges,adjacency)
        # output= m(output)
        loss = criterion(output, target)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self,batch,batch_idx):
        adjacency, nodes, edges, target = batch
        output = self.forward(nodes, edges,adjacency)
        loss = criterion(output, target)
        return {'val_loss':loss}

    def validation_epoch_end(self,outputs):
        avg_loss=torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs={'val_loss':avg_loss}
        return {'avg_val_loss':avg_loss,'log':tensorboard_logs}

    def test_step(self, batch, batch_idx):
        adjacency, nodes, edges, target = batch
        output = self(nodes, edges,adjacency)
        #output=m(output)
        #output=(output>=0.5).float()

        #auroc = metric(output, target).numpy()
        #acc=accuracy(output, target)
        loss = criterion(output, target)
        r2 = metric(output, target)
        return {'test_r2': r2,'test_loss': loss}# {'test_auroc': auroc,'accu':acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        #avg_auroc = sum([x['test_auroc'] for x in outputs])/len([x['test_auroc'] for x in outputs])
        #avg_acc = sum([x['accu'] for x in outputs])/len([x['accu'] for x in outputs])
        #logs = {'test_auroc': avg_auroc}
        #print(avg_acc)
        avg_r2 = sum([x['test_r2'] for x in outputs])/len([x['test_r2'] for x in outputs])
        print("test_r2",avg_r2)
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs} #{'test_auroc': avg_auroc, 'log': logs}

    

#runner fn 
def run_emn(node_features=40, edge_features=4,edge_embedding_size=50, message_passes=8, out_features=1,
            edge_emb_depth=3, edge_emb_hidden_dim=150,
                 att_depth=3, att_hidden_dim=80,
                 msg_depth=3, msg_hidden_dim=80,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=100,
                 gather_emb_depth=3, gather_emb_hidden_dim=100,
                 out_depth=2, out_hidden_dim=100):
  
    model = EMNImplementation(node_features=node_features, edge_features=edge_features,edge_embedding_size=edge_embedding_size, message_passes=message_passes, out_features=out_features,
            edge_emb_depth=edge_emb_depth, edge_emb_hidden_dim=edge_emb_hidden_dim,
                 att_depth=att_depth, att_hidden_dim=att_hidden_dim,
                 msg_depth=msg_depth, msg_hidden_dim=att_hidden_dim,
                 gather_width=gather_width,
                 gather_att_depth=gather_att_depth, gather_att_hidden_dim=gather_att_hidden_dim,
                 gather_emb_depth=gather_emb_depth, gather_emb_hidden_dim=gather_att_hidden_dim,
                 out_depth=out_depth, out_hidden_dim=out_hidden_dim)
    trainer = pl.Trainer(max_epochs=250)
    trainer.fit(model)
    evaluation = trainer.test(model=model)
    return evaluation

bounds = [{'name': 'edge_embedding_size', 'type': 'discrete',  'domain': (30,50)},
          {'name': 'message_passes',          'type': 'discrete',  'domain': (4,6,8)},
          {'name': 'edge_emb_hidden_dim',          'type': 'discrete',  'domain': (80,120,150)},
          {'name': 'att_hidden_dim',           'type': 'discrete',    'domain': (60,80,100)},
          {'name': ' gather_width',           'type': 'discrete',    'domain': (60,80,100)},
          {'name': 'gather_att_hidden_dim',       'type': 'discrete',    'domain': (60,80,100)},
          {'name': 'out_hidden_dim',           'type': 'discrete',    'domain': (60,80,100)}]

#function to optimize
def f(x):
    print(x)
    evaluation = run_emn(
        edge_embedding_size = int(x[:,0]),
        message_passes = int(x[:,1]), 
        edge_emb_hidden_dim = int(x[:,2]), 
        att_hidden_dim = int(x[:,3]),
        gather_width = int(x[:,4]), 
        gather_att_hidden_dim = int(x[:,5]), 
        out_hidden_dim = int(x[:,6])
        )
    print("test_loss:\t{0}".format(evaluation['test_loss']))
    return evaluation['test_loss']

if __name__=='__main__':
   opt_emn = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds,acquisition_type='EI',evaluator_type='local_penalization',initial_design_numdata=3)
   opt_emn.run_optimization(max_iter=10,max_time=36000)
   opt_emn.save_evaluations("ev_file")
   print("""
   Optimized Parameters:
   \t{0}:\t{1}
   \t{2}:\t{3}
   \t{4}:\t{5}
   \t{6}:\t{7}
   \t{8}:\t{9}
   \t{10}:\t{11}
   \t{12}:\t{13}
   """.format(bounds[0]["name"],opt_emn.x_opt[0],
              bounds[1]["name"],opt_emn.x_opt[1],
              bounds[2]["name"],opt_emn.x_opt[2],
              bounds[3]["name"],opt_emn.x_opt[3],
              bounds[4]["name"],opt_emn.x_opt[4],
              bounds[5]["name"],opt_emn.x_opt[5],
              bounds[6]["name"],opt_emn.x_opt[6]))
   print("optimized loss: {0}".format(opt_emn.fx_opt))

