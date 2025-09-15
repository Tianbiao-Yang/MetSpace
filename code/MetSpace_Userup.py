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
from cairosvg import svg2png
import xlsxwriter
from PIL import Image
import matplotlib.colors as colors
import numpy as np




def result_excel(hsc_three, result_type):

    for k,v in hsc_three.items():
        mol = Chem.MolFromSmiles(v[0])
        try:
            drawer = rdMolDraw2D.MolDraw2DSVG(280,280)
            drawer.SetFontSize(1)
            op = drawer.drawOptions()

            mol = rdMolDraw2D.PrepareMolForDrawing(mol)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            svg2 = svg.replace('svg:','')
            with open('../data/svg_data/' + k + '.svg','w') as wpklf:
                wpklf.write(svg2) 
            svg2png(url='../data/svg_data/'+ k + '.svg', write_to='../data/svg_data/'+ k + '.png',dpi=300)
        except:
            print(v[0])
        
    hmdb_id,smi_list = list(),list()
    for k,v in hsc_three.items():
        hmdb_id.append(k)
        smi_list.append(v[0])
    data_three = {'accession':hmdb_id, 'smiles':smi_list}
    df_three = pd.DataFrame(data_three)

    path = r"../data/svg_data/"
    pics = os.listdir(path)
    # Define the names of Excel and worksheets to be written
    book = xlsxwriter.Workbook(r"../data/result_" + result_type  + ".xlsx")
    sheet = book.add_worksheet(result_type)

    # Define the name of the two columns, and then fill in the nicknames to match.
    cell_format = book.add_format({'bold':True, 'font_size':16,'font_name':'Times New Roman','align':'center'})
    cell_format_1 = book.add_format({'font_size':16,'font_name':'Times New Roman','align':'center','align':'center'})
    cell_format_1.set_align('center')
    cell_format_1.set_align('center')
    sheet.set_column('A:A', 20)
    sheet.set_column('C:C', 20)

    sheet.write("A1", "Index_ID",cell_format)
    sheet.write("B1", "Structure",cell_format)
    sheet.write("C1", "Smiles",cell_format)

    sheet.write_column(1, 0, df_three.accession.values.tolist(),cell_format_1)
    sheet.write_column(1, 2, df_three.smiles.values.tolist(),cell_format_1)



    # To fix the size of the image, the cell where the picture is inserted must also be resized
    image_width = 280
    image_height = 280
    cell_width = 42
    cell_height = 240
    sheet.set_column("B:B", cell_width) # Set cell column width
    for i in range(len(df_three.accession.values.tolist())):
        if df_three.accession.values.tolist()[i] + ".png" in pics:
            # Fixed width / width of the original picture to be inserted
            x_scale = image_width / (Image.open(os.path.join(path, df_three.accession.values.tolist()[i] + ".png")).size[0]) 
            # Fixed height / height of the original picture to be inserted
            y_scale = image_height / (Image.open(os.path.join(path, df_three.accession.values.tolist()[i] + ".png")).size[1]) 
            sheet.set_row(i + 1, cell_height) # Set the row height
            sheet.insert_image(
                "B{}".format(i + 2),
                os.path.join(path, df_three.accession.values.tolist()[i] + ".png"),
                {"x_scale": x_scale, "y_scale": y_scale, "x_offset": 15, "y_offset": 20},
            )  # Set the x_offset and y_offset so that the image is centered as much as possible
    sheet.set_zoom(zoom=50)
    book.close()
    # Remove the svg picture
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
    
    
    print('Finally, the result of ' + result_type + ' was finished !!!')
    
result_type = 'MetSpace_Dr.Yao'
MetBA = dict()
with open('../result/MetSpace_Dr.Yao.txt') as rpklf:
    for i_data in rpklf:
        i = i_data.rstrip('\n').split('\t')
        if i[0] == 'MS_ID':
            pass
        else:
            MetBA[i[0]] = [i[1],i[2]]
            
            
result_excel(MetBA, result_type)            