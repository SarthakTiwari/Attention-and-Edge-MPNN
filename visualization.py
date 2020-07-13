import pandas as pd
import numpy as np
import torch

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from io import BytesIO
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions

from captum.attr import IntegratedGradients

from preprocessing import smile_to_graph,molgraph_collate_fn
from  edge_memory_network import EMNImplementation

#model input -->  nodes, edges,adjacency, target 



def get_colors(attr, colormap):
    attr2=attr.sum(dim=1)
    vmin=-max(attr.abs().max(), 1e-16)
    vmax=max(attr.abs().max(), 1e-16)
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(attr2))

def moltopng(mol,node_colors, edge_colors, molSize=(450,150),kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],molSize[1])
    drawer.drawOptions().useBWAtomPalette()
    drawer.drawOptions().padding = .2
    drawer.DrawMolecule(
        mc,
        highlightAtoms=[i for i in range(len(node_colors))], 
        highlightAtomColors={i: tuple(c) for i, c in enumerate(node_colors)}, 
        highlightBonds=[i for i in range(len(edge_colors))],
        highlightBondColors={i: tuple(c) for i, c in enumerate(edge_colors)},
        highlightAtomRadii={i: .5 for i in range(len(node_colors))}
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def visualize_importances(feature_names, importances, title="Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(20,10))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names)#,rotation='vertical')
        plt.xlabel(axis_title)
        plt.title(title)

# make sure feature_list has same len as node feature len
def visualizations(model,smile,target_value,feature_list,color_map= plt.cm.bwr):
    
    model.eval()

    adjacency, nodes, edges = smile_to_graph(smile)
    targets=np.expand_dims(target_value, axis=0)

    mols = Chem.MolFromSmiles(smile)
    ig = IntegratedGradients(model)
    adjacency, nodes, edges, targets=molgraph_collate_fn(np.expand_dims(((adjacency, nodes, edges), targets), axis=0))  # input dimension --> no. of smiles *smiles 
    attr= ig.attribute(nodes,additional_forward_args= (edges,adjacency),target=0 )

    attr1=torch.squeeze(attr, dim=0)
    attr2=attr1.sum(dim=1)
    vmax = max(attr2.abs().max(), 1e-16)
    vmin = -vmax

    node_colors = get_colors(attr1, color_map)
    node_colors=node_colors[:,:3]

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = plt.Normalize(vmin, vmax)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map),cax=ax, orientation='horizontal', label='color_bar')

    b = BytesIO(); b.write(moltopng(mols, node_colors=node_colors, edge_colors={}, molSize=(600,600))); b.seek(0)
    display(Image.open(b))
    b.close()

    symbols= {i: f'{mols.GetAtomWithIdx(i).GetSymbol()}{i}' for i in range(mols.GetNumAtoms())}


    x_pos = (np.arange(len(feature_list)))
    y_pos = (np.arange(len(list(symbols.values()))))
    plt.matshow(attr1,cmap=color_map)
    plt.xticks(x_pos,feature_list,rotation='vertical')
    plt.yticks(y_pos,list(symbols.values()))
    plt.show()

    visualize_importances(list(symbols.values()), attr2)



if __name__=='__main__':
  #Dataset on which model is trained
  data = pd.read_csv('sider.csv' )

  index=1104                                #change index to get molecule specific visualisations 
  target='Hepatobiliary disorders'          # target fixed for a perticular pretrained model

  smile=data['smiles'][index]
  target_value=data[target][index]

  feature_list=[
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Unknown',
        'Degree',
        'ImplicitValence',
        'FormalCharge',
        'NumRadicalElectrons',
        'IsAromatic',
        'TotalNumHs']
    
  #Loading pretrained model 
  model = EMNImplementation(node_features=16, edge_features=4,edge_embedding_size=25, message_passes=4, out_features=1)
  checkpoint = torch.load(r"checkpoint.ckpt")
  model.load_state_dict(checkpoint['state_dict'])
  

  # function which returns :-
  # molecule visualization with highlighted atoms
  # nodes vs feature matrix visualization
  # nodes attribution score
  visualizations(model,smile,target_value,feature_list,color_map= plt.cm.bwr)
