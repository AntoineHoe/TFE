# YOLO3 (Detection, Training, and Evaluation)
MIT License

Copyright (c) 2017 Ngoc Anh Huynh

For any question (except the data augmentation), check first the issues on the original repository : https://github.com/experiencor/keras-yolo3/
## Todo list:
- [x] Yolo3 detection
- [x] Yolo3 training (warmup and multi-scale)
- [x] mAP Evaluation
- [x] Multi-GPU training
- [x] Evaluation on VOC
- [x] Data Augmentation

## Installing

To install the dependencies, run
```bash
pip install -r requirements.txt
```
And for the GPU to work, make sure you've got the drivers installed beforehand (CUDA).

It has been tested to work with Python 2.7.13 and 3.5.3.

## Detection

Grab the pretrained weights of yolo3 from https://pjreddie.com/media/files/yolov3.weights.

```python yolo3_one_file_to_detect_them_all.py -w yolo3.weights -i dog.jpg``` 

## Training

### 1. Data preparation 

Download the Raccoon dataset from from https://github.com/experiencor/raccoon_dataset.

Organize the dataset into 4 folders:

+ train_image_folder <= the folder that contains the train images.

+ train_annot_folder <= the folder that contains the train annotations in VOC format.

+ valid_image_folder <= the folder that contains the validation images.

+ valid_annot_folder <= the folder that contains the validation annotations in VOC format.
    
There is a one-to-one correspondence by file name between images and annotations. If the validation set is empty, the training set will be automatically splitted into the training set and validation set using the ratio of 0.8.

Also, if you've got the dataset split into 2 folders such as one for images and the other one for annotations and you need to set a custom size for the validation set, use `create_validation_set.sh` script to that. The script expects the following parameters in the following order:
```bash
./create_validation_set.sh $param1 $param2 $param3 $param4
# 1st param - folder where the images are found
# 2nd param - folder where the annotations are found
# 3rd param - number of random choices (aka the size of the validation set in absolute value)
# 4th param - folder where validation images/annots end up (must have images/annots folders inside the given directory as the 4th param)
```
### 2. Data Augmentation (XML files only)

Create a "vanilla_dataset_annot" where you put all of your XML files.
Create a "vanilla_dataset_img" where you put all of your images.

These two folder will be the bedrock of the data augmentation : every data augmentation will be based on these folder.

create a "aug_images" and "aug_annot" folder : these two folders will be fill by the new created images and xml annotation.

```python augment_data.py -n 2 -l 0```

With this command line, you will create 2 augmented image based on only 1 image from the vanilla dataset.
The second parameter is the number of previous data augmentation you did.

The new augmented images and annotation files will be copy to the train folders.

If you realize it is not enough, you can create more augmented images with.

```python augment_data.py -n 10 -l 2```
 
As you already created 2 data augmentation, you have to set the seconde parameter to 2.
That way, it will create the 3 to 12 generation of data augmentation.
If you don't you may overwrite the previous data augmentation.
After the data augmentation, the augmented images and XML files will be copy to the train_image and train_annot folders

### 3. Edit the configuration file
The configuration file is a json file, which looks like this:

```python
{
    "model" : {
        "min_input_size":       288,
        "max_input_size":       448,
        "anchors":              [24,33, 34,100, 49,268, 88,39, 98,123, 144,280, 239,56, 291,152, 354,359],
        "labels":               ["Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"]
    },

    "train": {
        "train_image_folder":   "path\\to\\train_image_folder\\",
        "train_annot_folder":   "path\\to\\train_annot_folder\\",
        "cache_name":           "defect_aug_train.pkl",

        "train_times":          8,
        "batch_size":           2,
        "learning_rate":        1e-4,
        "nb_epochs":            100,
        "warmup_epochs":        3,
        "ignore_thresh":        0.5,
        "gpus":                 "0",

        "grid_scales":          [1,1,1],
        "obj_scale":            5,
        "noobj_scale":          1,
        "xywh_scale":           1,
        "class_scale":          1,

        "tensorboard_dir":      "logs",
        "saved_weights_name":   "defect.h5",
        "debug":                true
    },

    "valid": {
        "valid_image_folder":   "",
        "valid_annot_folder":   "",
        "cache_name":           "",

        "valid_times":          1
    }
}


```

The ```labels``` setting lists the labels to be trained on. Only images, which has labels being listed, are fed to the network. The rest images are simply ignored. By this way, a Dog Detector can easily be trained using VOC or COCO dataset by setting ```labels``` to ```['dog']```.

Download pretrained weights for backend at:

https://bit.ly/39rLNoE

**This weights must be put in the root folder of the repository. They are the pretrained weights for the backend only and will be loaded during model creation. The code does not work without this weights.**

### 4. Generate anchors for your dataset (optional)

`python gen_anchors.py -c config.json`

Copy the generated anchors printed on the terminal to the ```anchors``` setting in ```config.json```.

### 5. Start the training process

`python train.py -c config.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

### 6. Perform detection using trained weights on image, set of images, video, or webcam
`python predict.py -c config.json -i /path/to/image/or/video`

It carries out detection on the image and write the image with detected bounding boxes to the same folder.

If you wish to change the object threshold or IOU threshold, you can do it by altering `obj_thresh` and `nms_thresh` variables. By default, they are set to `0.5` and `0.45` respectively.

## Evaluation

`python evaluate.py -c config.json`

Compute the mAP performance of the model defined in `saved_weights_name` on the validation dataset defined in `valid_image_folder` and `valid_annot_folder`.
