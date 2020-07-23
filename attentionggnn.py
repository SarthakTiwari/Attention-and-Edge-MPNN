import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from preprocessing import smile_to_graph,molgraph_collate_fn
from modules import GraphGather, FeedForwardNetwork
from attention_aggregation import AggregationMPNN
from torch.utils.data import DataLoader
criterion = nn.MSELoss()
from sklearn.metrics import r2_score
metric = r2_score
import GPyOpt

class AttentionGGNN(AggregationMPNN):

    def __init__(self, node_features, edge_features, message_size, message_passes, out_features,
                 msg_depth=4, msg_hidden_dim=200, msg_dropout_p=0.0,
                 att_depth=3, att_hidden_dim=200, att_dropout_p=0,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=100, gather_att_dropout_p=0.0,
                 gather_emb_depth=3, gather_emb_hidden_dim=100, gather_emb_dropout_p=0.0,
                 out_depth=2, out_hidden_dim=100, out_dropout_p=0.0, out_layer_shrinkage=1.0):
        super(AttentionGGNN, self).__init__(node_features, edge_features, message_size, message_passes, out_features)

        self.msg_nns = nn.ModuleList()
        self.att_nns = nn.ModuleList()
        for _ in range(edge_features):
            self.msg_nns.append(
                FeedForwardNetwork(node_features, [msg_hidden_dim] * msg_depth, message_size, dropout_p=msg_dropout_p, bias=False)
            )
            self.att_nns.append(
                FeedForwardNetwork(node_features, [att_hidden_dim] * att_depth, message_size, dropout_p=att_dropout_p, bias=False)
            )
        self.gru = nn.GRUCell(input_size=message_size, hidden_size=node_features, bias=False)
        self.gather = GraphGather(
            node_features, gather_width,
            gather_att_depth, gather_att_hidden_dim, gather_att_dropout_p,
            gather_emb_depth, gather_emb_hidden_dim, gather_emb_dropout_p
        )
        out_layer_sizes = [  
            round(out_hidden_dim * (out_layer_shrinkage ** (i / (out_depth - 1 + 1e-9)))) for i in range(out_depth)
        ]
        self.out_nn = FeedForwardNetwork(gather_width, out_layer_sizes, out_features, dropout_p=out_dropout_p)

    def aggregate_message(self, nodes, node_neighbours, edges, node_neighbour_mask):
        energy_mask = (node_neighbour_mask == 0).float() * 1e6

        embeddings_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.msg_nns[i](node_neighbours) for i in range(self.edge_features)
        ]
        embedding = sum(embeddings_masked_per_edge)
        energies_masked_per_edge = [
            edges[:, :, i].unsqueeze(-1) * self.att_nns[i](node_neighbours) for i in range(self.edge_features)
        ]
        energies = sum(energies_masked_per_edge) - energy_mask.unsqueeze(-1)
        attention = torch.softmax(energies, dim=1)
        return torch.sum(attention * embedding, dim=1)

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

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
        self.attggnn_train=tuple(data_list1[:int(l*0.8)])
        self.attggnn_val=tuple(data_list1[int(l*0.8):int(l*0.9)])
        self.attggnn_test=tuple(data_list1[int(l*0.9):])


    def train_dataloader(self):
        return DataLoader(self.attggnn_train, batch_size=50, shuffle=True, collate_fn=molgraph_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.attggnn_val, batch_size=10, shuffle=False, collate_fn=molgraph_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.attggnn_test, batch_size=10, shuffle=False, collate_fn=molgraph_collate_fn)
      
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.5e-4)
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
        print("test_loss",avg_loss)
        logs = {'test_r2': avg_r2}
        return {'test_r2': avg_r2, 'log': logs} #{'test_auroc': avg_auroc, 'log': logs}
    
#runner fn 
def run_attnggnn(node_features=16, edge_features=4, message_size=10, message_passes=4, out_features=1,
                 msg_depth=2, msg_hidden_dim=40,
                 att_depth=2, att_hidden_dim=40,
                 gather_width=40,
                 gather_att_depth=2, gather_att_hidden_dim=30, 
                 gather_emb_depth=2, gather_emb_hidden_dim=30, 
                 out_depth=2, out_hidden_dim=40):
  
    model = AttentionGGNN(node_features=node_features, edge_features=edge_features, message_size=message_size, message_passes=message_passes, out_features=out_features,
                 msg_depth=msg_depth, msg_hidden_dim=msg_hidden_dim,
                 att_depth=msg_depth, att_hidden_dim=msg_hidden_dim,
                 gather_width=gather_width,
                 gather_att_depth=gather_att_depth, gather_att_hidden_dim=gather_att_hidden_dim, 
                 gather_emb_depth=gather_att_depth, gather_emb_hidden_dim=gather_att_hidden_dim, 
                 out_depth=out_depth, out_hidden_dim=out_hidden_dim)
    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(model)
    evaluation = trainer.test(model=model)
    return evaluation


bounds = [{'name': 'message_size', 'type': 'discrete',  'domain': (10,20,30)},
          {'name': 'message_passes',          'type': 'discrete',  'domain': (4,6,8)},
          {'name': 'msg_hidden_dim',           'type': 'discrete',    'domain': (30,40,60)},
          {'name': ' gather_width',           'type': 'discrete',    'domain': (30,40,60)},
          {'name': 'gather_att_hidden_dim',       'type': 'discrete',    'domain': (20,30,40)},
          {'name': 'out_hidden_dim',           'type': 'discrete',    'domain': (30,40,60)}]


#function to optimize
def f(x):
    print(x)
    evaluation = run_attnggnn(
        message_size = int(x[:,0]),
        message_passes = int(x[:,1]), 
        msg_hidden_dim = int(x[:,2]), 
        gather_width = int(x[:,3]), 
        gather_att_hidden_dim = int(x[:,4]), 
        out_hidden_dim = int(x[:,5]))
    print("r2_score:\t{0}".format(evaluation['test_r2']))
    return evaluation['test_r2']


if __name__=='__main__':
   opt_attnggnn = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds,acquisition_type='EI',evaluator_type='local_penalization', maximize=True)
   opt_attnggnn.run_optimization(max_iter=15)
   opt_attnggnn.save_evaluations("ev_file")
   print("""
   Optimized Parameters:
   \t{0}:\t{1}
   \t{2}:\t{3}
   \t{4}:\t{5}
   \t{6}:\t{7}
   \t{8}:\t{9}
   \t{10}:\t{11}

   """.format(bounds[0]["name"],opt_attnggnn.x_opt[0],
              bounds[1]["name"],opt_attnggnn.x_opt[1],
              bounds[2]["name"],opt_attnggnn.x_opt[2],
              bounds[3]["name"],opt_attnggnn.x_opt[3],
              bounds[4]["name"],opt_attnggnn.x_opt[4],
              bounds[5]["name"],opt_attnggnn.x_opt[5],))
   print("optimized r2: {0}".format(opt_attnggnn.fx_opt))
