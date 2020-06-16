import os
import argparse
import pandas as pd
import glob
import xml.etree.ElementTree as ET

def makedirs(path):
    print("creating {} folder".format(path[:-1]))
    try:
        os.makedirs(path)
    except FileExistsError:
        print("Folder already created")
    except OSError:
        if not os.path.isdir(path):
            raise

def xml_builder(data, dest):
    
    folder = "csv"
    filename = data[0][0]
    width = data[0][1]
    height = data[0][2]
    depth = 3
    #building the general information tree
    annotations = ET.Element("annotation")
    file_name = ET.SubElement(annotations, "filename")
    folder_ = ET.SubElement(annotations, "folder")
    size = ET.SubElement(annotations, "size")
    width_ = ET.SubElement(size,"width")
    height_ = ET.SubElement(size,"height")
    depth_ = ET.SubElement(size,"depth")
    #adding the data
    file_name.text = filename
    folder_.text = folder
    width_.text = str(width)
    height_.text = str(height)
    depth_.text = str(depth)
    #building each object
    for i in range(len(data)):
        
        object_ = ET.SubElement(annotations, "object")
        defect_name = ET.SubElement(object_, "name")
        difficult_ = ET.SubElement(object_, "difficult")
        bndbox_ = ET.SubElement(object_, "bndbox")
        xmin_ = ET.SubElement(bndbox_, "xmin")
        ymin_ = ET.SubElement(bndbox_, "ymin")
        xmax_ = ET.SubElement(bndbox_, "xmax")
        ymax_ = ET.SubElement(bndbox_, "ymax")

        defect_name.text = data[i][3]
        difficult_.text = str(0)
        xmin_.text = str(round(data[i][4]))
        ymin_.text = str(round(data[i][5]))
        xmax_.text = str(round(data[i][6]))
        ymax_.text = str(round(data[i][7]))
    
    xml_name_file = "{}.xml".format(file_name.text[:-4])
    tree = ET.ElementTree(annotations)
    tree.write("{}{}".format(dest, xml_name_file))

def csv_to_xml(df, path):
    aug_bbs_xy = pd.DataFrame(columns=
                            ['filename','width','height','name', 'xmin', 'ymin', 'xmax', 'ymax']
                            )
    grouped = df.groupby('filename')
    for filename in df['filename'].unique():
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1) 
        group_df = group_df.dropna() 
        data = group_df.values
        xml_builder(data, path)

def _main_(args) :
    
    path = args.path_of_the_cvs_file
    dest = args.path_of_the_xml_folder

    makedirs(dest)

    for file_ in glob.glob(path+"\\*.csv"):

        df = pd.read_csv(file_)
        csv_to_xml(df, dest)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Convert cvs file to xml')

    argparser.add_argument('-c', '--path_of_the_cvs_file', help='path to the folder containing the csv file to convert to XML')
    argparser.add_argument('-x', '--path_of_the_xml_folder', default='XML_annot_folde\\', help='path of the new XML annotation folder')   

    args = argparser.parse_args()
    _main_(args)