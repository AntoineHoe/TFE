
# thanks to this reddit post : https://www.reddit.com/r/learnpython/comments/4ury67/elementtree_and_deeply_nested_xml/

import imgaug as ia
ia.seed(1)
# imgaug uses matplotlib backend for displaying images
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
import imgaug
# imageio library will be used for image input/output
import imageio
import pandas as pd
import numpy as np
import re
import glob
# this library is needed to read XML files for converting it into CSV
import xml.etree.ElementTree as ET
import shutil

def xml_builder(data):
    
    folder = "aug_img"
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
    tree.write("aug_annot/{}".format(xml_name_file))

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
        xml_builder(data)

def bbs_obj_to_df(bbs_object):
#     convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
#     convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs

def image_aug(df, images_path, aug_images_path, image_prefix, augmentor):
    defect_augmented_img =[]
    #print(augmentor)
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename','width','height','name', 'xmin', 'ymin', 'xmax', 'ymax']
                             )
    grouped = df.groupby('filename')
    
    for filename in df['filename'].unique():
    #   get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)  
    #   read the image
        image = imageio.imread(images_path+filename)
    #   get bounding boxes coordinates and write into array        
        bb_array = group_df.drop(['filename', 'width', 'height', 'name'], axis=1).values
        #print(bb_array)
    #   pass the array of bounding boxes coordinates to the imgaug library
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
    #   apply augmentation on image and on the bounding boxes
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
        #print(bbs_aug)
    #   disregard bounding boxes which have fallen out of image pane    
        bbs_aug = bbs_aug.remove_out_of_image()
    #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()
        
    #   don't perform any actions with the image if there are no bounding boxes left in it    
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            print("{} met an issue".format(filename))
            defect_augmented_img.append(filename)
            pass
        
    #   otherwise continue
        else:
        #   write augmented image to a file
            imageio.imwrite(aug_images_path+image_prefix+filename, image_aug)  
        #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)    
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
        #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix+x)
        #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
        #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
        #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])            
            #print("{} augmented".format(filename))
    # return dataframe with updated images and bounding boxes annotations 
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    print("done")
    print("{} augmented images have reach an issue :".format(len(defect_augmented_img)))
    for element in defect_augmented_img:
        print(element)

    return aug_bbs_xy

def parser(data):
    #list of every element we care in the xml files
    column_name = ['filename', 'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax']

    tree = ET.iterparse(data)
    #select all element inside the xml file
    for event, node in tree:
        #select the information in the list above and return them
        if node.tag in column_name:
            yield node.tag, node.text

def xml_to_csv(path):
    xml_list = []
    column_name = ['filename', 'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax']
    defect_list = ["Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"]
    for xml_file in glob.glob(path + '/*.xml'):
        # empty all the list for each new xml file
        basic_img_info =[]
        fin =[]
        img = []
        new_img = []
        with open(xml_file, 'r') as myFile:

            results = parser(myFile)
            #all the information we need is add in one list
            for tag, text in results:
                img.append(text)
        #print(img)

        #the img list looks like this : 

        # ['filename', 'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax',  'name' ,[...] 'ymin', 'xmax', 'ymax']

        # with a repeted sequence ['name', 'xmin', 'ymin', 'xmax', 'ymax'] for each defect contained in the xml file


        #if the list contain more than 1 defect
        while len(img) > 8:

            #we selected the last 5 data of the big list: ['name', 'xmin', 'ymin', 'xmax', 'ymax']

            #and add it to the basic image info : ['filename', 'width', 'height']

            #then we delete the last 5 data of the big list

            # and we go on again and again

            #until the list contain only 1 defect : ['filename', 'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax']

            basic_img_info = img[0:3]

            fin = img[-5 :]
            
            new_img = basic_img_info + fin
            xml_list.append(new_img)
            del img[-5:]

        xml_list.append(img)
        # for element in xml_list :
        #     print(element)
    print(len(xml_list)) #8323
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    
    return xml_df

aug = iaa.SomeOf(2, [    
    # iaa.Affine(scale=(0.5, 1.5)),
    # iaa.Affine(rotate=(-30, 30)),
    # iaa.Affine(translate_percent={"x":(-0.2, 0.2),"y":(-0.2, 0.2)}),
    iaa.Fliplr(1),
    iaa.SaltAndPepper(0.1, per_channel=True),
    iaa.Solarize(0.5, threshold=(32, 128)),
    iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add((0, 50))))
    
])

labels_df = xml_to_csv('train_image_folder/')
labels_df.to_csv(('labels.csv'), index=None)

augmented_images_df = image_aug(labels_df, 'train_image_folder/', 'aug_images/', 'aug1_', aug)

#create a next generation of augmented img
#augmented_images_df = image_aug(labels_df, 'train_image_folder/', 'aug_images/', 'aug2_', aug)


# Concat resized_images_df and augmented_images_df together and save in a new all_labels.csv file
all_labels_df = pd.concat([labels_df, augmented_images_df])
all_labels_df.to_csv('all_labels.csv', index=False)
csv_to_xml(augmented_images_df, 'aug_images/')

# Lastly we can copy all our augmented images in the same folder as original resized images
# for file in os.listdir('aug_images'):
#     shutil.copy('aug_images/'+file, 'train_image_folder/'+file)