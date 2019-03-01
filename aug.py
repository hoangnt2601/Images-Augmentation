import cv2
import numpy as np
import os.path
import errno
import color_aug as ca
import trans_aug as trans

dataset_dir="/home/hoangnt/workspace/Project/DL/License_Plate_Recognition/test/"
save_dir="/home/hoangnt/workspace/Project/DL/License_Plate_Recognition/auged/"
extension=".jpeg"


class IdentityMetadata():
    def __init__(self, base, file):
        # dataset base directory
        self.base = base
        # image file name
        self.file = file
    def __repr__(self):
        return self.image_path()
    def image_path(self):
        return os.path.join(self.base, self.file) 
    
class IdentityLabels():
    def __init__(self, img_name):
        # image file name
        self.img_name = img_name
    def __repr__(self):
        return self.label()
    def label(self):
        return os.path.join(self.img_name) 
    
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        metadata.append(IdentityMetadata(path, i))
    return np.array(metadata)

def load_label(path):
    label = []
    for i in os.listdir(path):
        label.append(IdentityLabels(i.split('.')[0]))
    return np.asarray(label, dtype=np.str)

def create_folder(path):
    # create folder with label
    #print(len(load_label(dataset_dir)))
    for i in range(len(load_label(dataset_dir))):
        filename = save_dir + load_label(dataset_dir)[i]
        #print(filename)
        if not os.path.exists(filename):
            try:
                os.mkdir(filename)
                #print(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
'''
create_folder(save_dir)
'''

metadata = load_metadata('test')
for i in range(0, metadata.size):
    dir_to_save_label = save_dir + load_label(dataset_dir)[i]
    img = cv2.imread(str(metadata[i]))
    img = trans.rotate_random_image(img)
    cv2.imwrite(os.path.join(dir_to_save_label , 'rotate_random5-' + load_label(dataset_dir)[i] + extension),img)