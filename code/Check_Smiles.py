import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import rdkit
from rdkit import Chem
import pandas as pd
from rdkit.Chem import rdMolDescriptors, MolSurf
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D, MolToFile, _moltoimg
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# %matplotlib inline
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.image as mpimg
# from IPython.display import SVG, display
import seaborn as sns
from PIL import Image
import matplotlib.colors as colors
import numpy as np

ex_test = dict()
with open('../data/BILELIB19_LIPIDMAPS.txt') as rpklf:
    for i_data in rpklf:
        i = i_data.rstrip('\n').split('\t')
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(i[0]))
        ex_test[smi] = ''


ba_set = dict()
with open('../data/BAs_set.txt') as rpklf:
    for i_data in rpklf:
        i = i_data.rstrip('\n').split('\t')
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(i[0]))
        ba_set[smi] = i[0]

dep = dict()
for k,v in ex_test.items():
    if k in ba_set:
        # print(k,ex_test[k])
        dep[k] = v
print('The number of repeated molecular structures: ',len(dep))
