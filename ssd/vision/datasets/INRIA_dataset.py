import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os


class INRIA_Dataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None , split = None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        # self.root = pathlib.Path(root)
        self.root = root
        data_dir = root
        self.transform = transform
        self.target_transform = target_transform
        
        
        if is_test:
            subset = 'Test'
            id_list_file = os.path.join(data_dir,subset,'annotations.lst')

            # image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            subset = 'Train'
            id_list_file = os.path.join(data_dir,subset,'annotations.lst')
            # image_sets_file = self.root / "ImageSets/Main/trainval.txt"        
        
        self.subset = subset
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        
        if split is not  None:
            if split == 'left':
                print('split left')
                self.ids = self.ids[: int ( len(self.ids) //2) ]
            elif split == 'right':
                print('split right')
                self.ids = self.ids[  int ( len(self.ids) //2):]
            else:
                raise ZeroDivisionError
        else:
            print('do not split dataset')
            
            
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        # label_file_name = self.root / "labels.txt"

        if False:#os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list
            
            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default INRIA classes.")
            self.class_names = ('BACKGROUND',
            'person',)


        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        # print(index)
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(index)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index].replace('.txt','.png')
        img_path = image_id.replace('annotations','pos')
        img_path = os.path.join(self.root,img_path)

        # print(img_path)
        image = self._read_image(img_path)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(index)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
    
        id_ = self.ids[image_id]
        boxes_list=[]
        labels_list=[]
        #with open(id_,encoding='ISO-8859-1') as f:
        annot_path = os.path.join(self.root,id_)
        with open(annot_path , encoding = "ISO-8859-1") as f:
            a= f.readlines()
        box_found = False
        label_found = False
        boxes = []
        labels = []
        is_difficult=[]
        for l in a:
            word = 'Image size (X x Y x C) : '
            if word in l:
                info = l[l.find(word)+len(word):]
                info = info.strip()

            word = 'label for object'
            if word in l:
                key = ':'
                info =l[l.find(key)+len(key):]
                info = info.replace('"','')
                info = info.strip()
                label = info
                if label =='UprightPerson':
                    label = 1
                else:
                    print('another label : {}'.format(label))
                labels.append(label)
                is_difficult.append(0)
                label_found =True
            word = 'Bounding box for object'
            if word in l:
                key  = '(Xmin, Ymin) - (Xmax, Ymax) :'
                info = l[l.find(key)+len(key):]
                info = info.strip()
                info = info.replace('(','')
                info = info.replace(')','')
                info = info.replace('-','')
                info = info.replace(',','')
                info = info.split()
                box  = [int(item) for item in info]
                box  = np.array([box[1],box[0],box[3],box[2]]) #xy  conversion
                boxes.append(box)
                box_found = True 
        if not box_found:
            print('box not found')
        if not label_found:
            print('label not found')
        bbox = boxes
        bbox  = np.stack(bbox).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)
        # return bbox, labels
        
        return (np.array(bbox, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    
    
     

    def _read_image(self, image_id):
        image_id = image_id.replace('.txt','.png')
        img_path = image_id.replace('annotations','pos')
        img_path = os.path.join(self.root,img_path)
        
        image_file =img_path

        # print(image_file)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VOCDataset_split:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None , split = None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
            
        print(image_sets_file)
        self.ids = VOCDataset_split._read_image_ids(image_sets_file)
        if split is not  None:
            if split == 'left':
                self.ids = self.ids[: int ( len(self.ids) //2) ]
            elif split == 'right':
                self.ids = self.ids[  int ( len(self.ids) //2):]
            else:
                raise ZeroDivisionError
        else:
            print('no split dataset')

        
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list
            
            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')


        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        
        print( 'len of ids' , len(self.ids))

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        # print(image_id)
        # print(image_file)
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
