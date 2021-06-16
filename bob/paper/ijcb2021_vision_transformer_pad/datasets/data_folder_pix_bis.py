#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ageorge
"""

#==============================================================================
# Import what is needed here:

import torch.utils.data as data
import os

import random
random.seed( a = 7 )

import numpy as np

from random import shuffle

import random

import bob.io.image

import h5py

def get_frame(fname,frame_idx=1):

    f = h5py.File(fname, 'r')
    path='Frame_'+str(frame_idx)+'/array'
    data=np.array(f.get(path))    
    return data


def get_key_indexes(fname):

    f = h5py.File(fname, 'r')
    keys=list(f.keys())
    indexes=[int(ff.split("_")[-1]) for ff in keys if ff!='FrameIndexes']

    return indexes


def balance_data_filebased(fnil):
    from collections import Counter
    
    ''' Return a balanced list of files tuples '''

    # print('fnil',fnil)
    reals=[ff for ff in fnil if ff[2]==0]
    
    attacks=[ff for ff in fnil if ff[2]==1]
    
    num_reals=len(reals)
    num_attacks=len(attacks)
    
    print("num_reals",len(reals),"num_attacks",len(attacks))
    
    ## Simplest, always downsample, no upsampling here, from each file select a number of frames so as to balance with the other class. 
    
    if num_reals>num_attacks:

        ## downsample reals
        
        names=[ff[0] for ff in reals]
        
        cnt=Counter(names)
        
        frames_perfile=round(num_attacks/len(cnt)) # would be low, now add some more to solve the issue
        
        new_reals=[]
        new_attacks=attacks
        
        
        for file_name in sorted(cnt.keys()):
            
            num_files=cnt[file_name]
            
            try:
                sample_idx=random.sample(range(num_files),frames_perfile)
            except:
                sample_idx=range(num_files)
            
            ffs=[ff for ff in reals if ff[0]==file_name]
            
            ffs=[ff for idx,ff in enumerate(ffs) if idx in sample_idx]
            new_reals=new_reals+ffs
            
            
        try:
            new_indexes=random.sample(range(len(reals)),len(new_attacks)-len(new_reals))
        except:
            new_indexes=[]
        
        new_reals=new_reals+[ff for idx,ff in enumerate(reals) if idx in new_indexes]
        
        
        fnil_balanced=new_reals+new_attacks
               
        
    else:
        # downsample attacks        
        names=[ff[0] for ff in attacks]
        
        cnt=Counter(names)
        
        frames_perfile=round(num_reals/len(cnt)) # would be low, now add some more to solve the issue
        
        new_reals=reals
        new_attacks=[]
        
        for file_name in sorted(cnt.keys()):
            
            num_files=cnt[file_name]
            
            try:
                sample_idx=random.sample(range(num_files),frames_perfile)
            except:
                sample_idx=range(num_files)
            
            ffs=[ff for ff in attacks if ff[0]==file_name]
            
            ffs=[ff for idx,ff in enumerate(ffs) if idx in sample_idx]
            new_attacks=new_attacks+ffs
            
            
        try:
            new_indexes=random.sample(range(len(attacks)),len(new_reals)-len(new_attacks))
        except:
            new_indexes=[]
            
        
        new_attacks=new_attacks+[ff for idx,ff in enumerate(attacks) if idx in new_indexes]
        

        fnil_balanced=new_reals+new_attacks

    print("num_reals",len(new_reals),"num_attacks",len(new_attacks))

    return fnil_balanced


def get_index_from_file_names(file_names_and_labels, data_folder,max_samples_per_file):

    file_names_index_and_labels=[]

    for file_name, label in file_names_and_labels:

        try:

            indexes=get_key_indexes(os.path.join(data_folder,file_name)) ## fetches the frame indexes present in the preprocessed file
        except:
            indexes=[]

        # sample the frames from video

        if len(indexes)>=max_samples_per_file:
            indexes = random.sample(indexes, max_samples_per_file)        

        for k in indexes:

            file_names_index_and_labels.append( ( file_name, k, label ) )

    return file_names_index_and_labels


def get_file_names_and_labels(files, extension = ".hdf5", hldi_type = "fr",extra=None):
    """
    Get absolute names of the corresponding file objects and their class labels.

    **Parameters:**

    ``files`` : [File]
        A list of files objects defined in the High Level Database Interface
        of the particular datbase.

    ``data_folder`` : str
        A directory containing the training data.

    ``extension`` : str
        Extension of the data files. Default: ".hdf5" .

    ``hldi_type`` : str
        Type of the high level database interface. Default: "pad".
        Note: this is the only type supported at the moment.

    **Returns:**

    ``file_names_and_labels`` : [(str, int)]
        A list of tuples, where each tuple contain an absolute filename and
        a corresponding label of the class.
    """

    file_names_and_labels = []


    for f in files:

        if hldi_type == "pad":

            if f.attack_type is None:

                label = 1

            else:

                label = 0

        if extra is not None:
            file_names_and_labels.append( ( os.path.join(f.path + extra+  extension), label ) )
        else:
            file_names_and_labels.append( ( os.path.join(f.path +  extension), label ) )

    return file_names_and_labels


#==============================================================================
class DataFolderPixBiS(data.Dataset):
    """
    A generic data loader compatible with Bob High Level Database Interfaces
    (HLDI). Only HLDI's of bob.pad.face are currently supported.
    """

    def __init__(self, data_folder,
                 transform = None,
                 extension = '.hdf5',
                 bob_hldi_instance = None,
                 hldi_type = "pad",
                 groups = ['train', 'dev', 'eval'],
                 protocol = 'nowig',
                 purposes=['real', 'attack'],
                 allow_missing_files = True,
                 do_balance=False,
                 max_samples_per_file=1000,
                 channels='RGB',
                 mask_op='flat',
                 custom_size=224,
                 **kwargs):
        """
        **Parameters:**

        ``data_folder`` : str
            A directory containing the training data.

        ``transform`` : callable
            A function/transform that  takes in a PIL image, and returns a
            transformed version. E.g, ``transforms.RandomCrop``. Default: None.

        ``extension`` : str
            Extension of the data files. Default: ".hdf5".
            Note: this is the only extension supported at the moment.

        ``bob_hldi_instance`` : object
            An instance of the HLDI interface. Only HLDI's of bob.pad.face
            are currently supported.

        ``hldi_type`` : str
            String defining the type of the HLDI. Default: "pad".
            Note: this is the only option currently supported.

        ``groups`` : str or [str]
            The groups for which the clients should be returned.
            Usually, groups are one or more elements of ['train', 'dev', 'eval'].
            Default: ['train', 'dev', 'eval'].

        ``protocol`` : str
            The protocol for which the clients should be retrieved.
            Default: 'grandtest'.

        ``purposes`` : str or [str]
            The purposes for which File objects should be retrieved.
            Usually it is either 'real' or 'attack'.
            Default: ['real', 'attack'].

        ``allow_missing_files`` : str or [str]
            The missing files in the ``data_folder`` will not break the
            execution if set to True.
            Default: True.
        """

        self.data_folder = data_folder
        self.modalities = 'color'
        self.transform = transform
        self.extension = extension
        self.bob_hldi_instance = bob_hldi_instance
        self.hldi_type = hldi_type
        self.groups = groups
        self.protocol = protocol
        self.purposes = purposes
        self.allow_missing_files = allow_missing_files
        self.do_balance=do_balance
        self.max_samples_per_file=max_samples_per_file
        self.channels=channels
        self.mask_op=mask_op
        self.custom_size=custom_size
        
        print("SLEFHLDI",self.hldi_type)


        # if isinstance(bob_hldi_instance,bob.pad.face.database.ReplayMobilePadDatabase):
        #     extra="_replaymobile"
        # elif isinstance(bob_hldi_instance,bob.pad.face.database.ReplayPadDatabase):
        #     extra="_replay"
        # else:
        #     extra=None

        extra=None

        # print('bob_hldi_instance',bob_hldi_instance)

        if bob_hldi_instance is not None:

            files = bob_hldi_instance.objects(groups = self.groups,
                                              protocol = self.protocol,
                                              purposes = self.purposes,
                                              **kwargs)
            
            print("LENFILES",len(files))

            file_names_and_labels= get_file_names_and_labels(files = files, extension = self.extension,hldi_type = self.hldi_type,extra=extra)
                                            
                                            
            
            if self.allow_missing_files: # return only files which exists in all modalities

                file_names_and_labels_sel=[]


                for f in file_names_and_labels:
                    if all(os.path.isfile(os.path.join(data_folder,f[0]))==True for modality in self.modalities):
                        file_names_and_labels_sel.append(f)
                file_names_and_labels=file_names_and_labels_sel

    
        else:

            # TODO - add behaviour similar to image folder
            file_names_and_labels = []

        self.file_names_and_labels = file_names_and_labels # list of videos # relative paths, not absolute

        ## Added shuffling of the list

        file_names_index_and_labels=get_index_from_file_names(self.file_names_and_labels,self.data_folder,self.max_samples_per_file)




        ## Subsampling should be performed here. Should respect the attack types?, here it subsamples based on file
        # print('file_names_and_labels',file_names_and_labels)
        # print('file_names_index_and_labels',file_names_index_and_labels)
        # import pdb; pdb.set_trace()


        if self.do_balance:
            file_names_index_and_labels=balance_data_filebased(file_names_index_and_labels)
        
        random.shuffle(file_names_index_and_labels) 

        self.file_names_index_and_labels= file_names_index_and_labels


    #==========================================================================
    def __getitem__(self, index):

        # with index, figure out which file and which frame


        """
        Returns an image, possibly transformed, and a target class given index.

        **Parameters:**

        ``index`` : int.
            An index of the sample to return.

        **Returns:**

        ``pil_img`` : Tensor or PIL Image
            If ``self.transform`` is defined the output is the torch.Tensor,
            otherwise the output is an instance of the PIL.Image.Image class.

        ``target`` : int
            Index of the class.

        """

        try:

            stacker=[]
            for modality in ['color']:

                path,frame_idx, target = self.file_names_index_and_labels[index] # relative path

                modality_path = os.path.join(self.data_folder,path) # Now, absolute path to each of the files

                img_array=get_frame(modality_path,frame_idx=frame_idx)


                if img_array.shape[0]==3:

                    #  Even pixels in height and width

                    if img_array.shape[1]%2!=0:
                        img_array=img_array[:,0:-1,:]
                    if img_array.shape[2]%2!=0:
                        img_array=img_array[:,:,0:-1]


                    img_array_RGB=img_array.copy()

                    cv_RGB=bob.io.image.to_matplotlib(img_array_RGB)

                    img_array_tr=cv_RGB.copy()
                    
                else:
                    img_array_tr=img_array.copy()



                pil_img   = img_array_tr.copy()


                stacker.append(pil_img)

            stacker=pil_img.copy()
            stacker=np.array(stacker,dtype='uint8')## uint8 is required otherwise the PIL.Image wont know what it is


            #assert(img_array_RGB.shape[1]==224)

            pil_imgs_stacko=stacker.copy()


            if self.transform is not None:
                pil_imgs_stacko = self.transform(pil_imgs_stacko) 


            if target==1: # BF

                if self.mask_op=='flat':


                    mask=np.ones((img_array_RGB.shape[1],img_array_RGB.shape[2]),dtype='float')*0.99

                    if self.custom_size!=224:
                        mask=np.ones((self.custom_size,self.custom_size),dtype='float')*0.99

            else:
                mask=np.ones((img_array_RGB.shape[1],img_array_RGB.shape[2]),dtype='float')*0.01

                if self.custom_size!=224:
                    mask=np.ones((self.custom_size,self.custom_size),dtype='float')*0.01

            labels={}
            labels['pixel_mask']=mask
            labels['binary_target']=target

            img={}
            img['image']=pil_imgs_stacko

            return img,labels
        except Exception as e:
            print(e)


    #==========================================================================
    def __len__(self):
        """
        **Returns:**

        ``len`` : int
            The length of the file list.
        """
        return len(self.file_names_index_and_labels)




