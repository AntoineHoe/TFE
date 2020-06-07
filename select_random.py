import argparse
import os
import math
import random

def makedirs(path):
    print("creating {} folder".format(path[:-1]))
    try:
        os.makedirs(path)
    except FileExistsError:
        print("Folder already created")
    except OSError:
        if not os.path.isdir(path):
            raise

def _main_(args):

    #catch all the parameters
    ratio = args.ratio
    image_source = args.image_source
    annot_source = args.annot_source

    img_valid_dest_folder = "valid_image_folder\\"
    annot_valid_dest_folder = "valid_annot_folder\\"

    makedirs(img_valid_dest_folder)
    makedirs(annot_valid_dest_folder)

    #building the absolute path from the relative one
    absolut_path = os.path.dirname(__file__)

    image_source = os.path.join(absolut_path, image_source)
    annot_source = os.path.join(absolut_path, annot_source)

    image_valid_dest = os.path.join(absolut_path, img_valid_dest_folder)
    annot_valid_dest = os.path.join(absolut_path, annot_valid_dest_folder)


    # list of randomly selected img from the dataset
    listImg = []

    ratio = float(ratio)
    if ratio > 1:
        print('Invalid ratio : the ratio must be < 1')
        exit()

    # check if 
    # they have both the same number of files

    if len(os.listdir(image_source)) == len(os.listdir(annot_source)):

        #calculate the numbe of img/annot to transfer
        number_img = len(os.listdir(image_source))
        print("{} files in both train folders".format(number_img))

        number_file_to_separate = math.ceil(number_img * ratio)
        print("{} files will be move from train to valid".format(number_file_to_separate))

        random_file=random.choice(os.listdir(image_source+"\\"))
        listImg.append(random_file[:-4])
        print("The {} files will be move".format(random_file[:-4]))

    else:
        print('Image or annot folder does not have the same amount of files')

    #for the number of file to randomly select
    for i in range(number_file_to_separate):
        #select one random file
        random_file=random.choice(os.listdir(image_source+"\\"))

        random_file= random_file[:-4]

        print("The {} files will be move".format(random_file))

        #check if already in the list

        alReadyIn = False
        number_of_file_already_present = 0
        while not(alReadyIn):

            #select another file if already in the list
            if random_file in listImg:

                print("File {} already selected".format(random_file))
                print("Choosing a new file")
                random_file=random.choice(os.listdir(image_source+"\\"))
                random_file= random_file[:-4]
                number_of_file_already_present +=1
                
            # add the file if not currently in the list
            else:
                
                listImg.append(random_file)
                alReadyIn = True

    #move selected file to valid folders
    for i in listImg:
        image = i+".jpg"
        annot = i+".xml"
        os.rename(os.path.join(image_source, image),os.path.join(image_valid_dest, image))
        os.rename(os.path.join(annot_source, annot),os.path.join(annot_valid_dest, annot))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='randomise a full dataset to train and valid folder on any dataset')

    argparser.add_argument('-r', '--ratio', default='0.2', help='ratio of validation data, ideally 15-20%')   
    argparser.add_argument('-i', '--image_source', help='path of the image source folder') 
    argparser.add_argument('-a', '--annot_source', help='path of the annot source folder') 

    args = argparser.parse_args()
    _main_(args)
