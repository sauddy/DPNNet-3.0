# import the necessary packages

import pandas as pd
import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow import keras as k

print("We are currently using the Modules_DPCNet-RT")


def load_parameter_csv(folder_address):

    '''
    Reading the parameter file used in the FARGO3D sims##
    With each simulation there are 150 RT images after image augmentation

    '''
    #parameter_file = pd.read_csv(folder_address + 'cluster_run_1.csv')
    parameter_file = pd.read_csv(folder_address + 'cluster_run_1-700.csv')
    return parameter_file


def create_complete_data_csv(list_sorted_RT_path ,path):
    '''
    Input : list of address to the RT image folder, example "RT_A_1"
            Path to the working directory where the cluster_run_1.csv is and the data_folder is created
    
    Output : Return a conacted csv with sim parameters and path to the RT images 
             (optional)Returns csv for each simulations along with the path to the augmented images 
    
    '''
    
    dataset_complete = []
    print("[INFO]: Importing path for all the RT images")
    for index in range(len(list_sorted_RT_path)):

        path_image = list_sorted_RT_path[index] ## path to each RT folder

        ## for paths from the RT sim fodlers directly
        list_image_path = glob.glob(path_image + "/images/snu/"+ "image_"+"*.png") ## list of the path to each image in the RT folder
        if len(list_image_path) ==0 :## for updates image paths
            # print("Reading images from the updated folder")
            list_image_path = glob.glob(path_image+'/'+"*.png") ## list of the path to each image in the RT folder

        df_images_folder =pd.DataFrame(list_image_path,columns=["image_path"]) ## making a dataframe with the images path

       #df_images_folder.columns = ["image_path"]  ## columm name with image_path
        
        
        parameter_df = load_parameter_csv(path) ## loading the sim parameters csv file
        parameter_df['Planet_Mass1'] = parameter_df['Planet_Mass1'] / (3 * 10**-6)  # writing in unit of earth mass
        parameter_df['Planet_Mass2'] = parameter_df['Planet_Mass2'] / (3 * 10**-6)  # writing in unit of earth mass
        parameter_df['Planet_Mass3'] = parameter_df['Planet_Mass3'] / (3 * 10**-6)  # writing in unit of earth mass
        simulated_params = parameter_df.iloc[index:index+1,:] ## selecting each Fargo3D params 
        expanded_simulated_params = pd.concat([simulated_params]*150,ignore_index=True)  ## repeating 150 times for each image with same params

        dataset_each_sim = pd.concat([expanded_simulated_params,df_images_folder],axis=1) ## dataset for each Fargo3D sim
        
        ## if needed we can print the .csv for each simulations
#         dataset_each_sim.to_csv(path+'data_folder/RT_images'+str(index)+'.csv')  

        # Appending the data from the pandas dataframe for each orbits
        dataset_complete.append(dataset_each_sim)

    
    dataset_complete = pd.concat(dataset_complete, ignore_index=True, axis=0)
    dataset_complete.to_csv(path+'data_folder/dataset_complete.csv') 
    print("[INFO]: Contatination of the paths of all the RT images are now complete")
    return dataset_complete


def parse_dataset(dataset_path, filtering=True,drop=None):
    '''
    Input : Address to the data folder
    F,columns=["image_path"]iltering : To select the data
    Drop: If true it drops all feature except Aspect ratio
    Output : Return a csv with parameter data and path to the image

    '''

    dataset = load_csv(dataset_path, filtering=filtering,drop=drop)
    images = []
    for i in dataset["Sample#"]:
        imagePath = os.path.sep.join([dataset_path, "Disk_dust_plots/dust1_gap_{}.jpg".format(i)])
        images.append(imagePath)
    dataset["file"] = images
    return dataset

def process_the_disk_attributes(train, test, val, path,multi_label=None):

    ''' Input : train or test or val dataset
        path : to store the stats file
        Output : Return normalized data z normalization is used
    '''

    print("[INFO] preparing the normalized data training/testing split...")
    try: 
        print("Droping the irrelevant columns ")
        train  = train.drop(columns=['image_path']) ## dropping the necessary files
        test   = test.drop(columns=['image_path']) ## dropping the necessary files
        val   = val.drop(columns=['image_path']) ## dropping the necessary files
        # train  = train.drop(columns=['Sample#', 'image_path']) ## dropping the necessary files
        # test   = test.drop(columns=['Sample#', 'image_path']) ## dropping the necessary files
    except KeyError :
        pass
    
    if multi_label == None:
      print("INFO: considering only planet1 as output")
      train_stats = train.describe()
      train_stats.pop("Planet_Mass1")
      train_stats.pop("Planet_Count")
      train_stats = train_stats.transpose()

      train_stats.to_csv(path+'data_folder/train_stats.csv')

      ## The labels are not normalized
      train_labels = train.pop("Planet_Mass1")
      test_labels = test.pop("Planet_Mass1")
      val_labels = val.pop("Planet_Mass1")
        
      Y_train = train.pop("Planet_Count")
      Y_val = val.pop("Planet_Count")
      Y_test = test.pop("Planet_Count")
    
    
    else:
      print("INFO: considering multiple planets as output")
      train_stats = train.describe()
      train_stats.pop("Planet_Mass1")
      train_stats.pop("Planet_Mass2")
      train_stats.pop("Planet_Mass3")
      train_stats.pop("Planet_Count")
      train_stats = train_stats.transpose()

      
      train_stats.to_csv(path+'data_folder/train_stats_multi.csv')


      train_labels = train[["Planet_Mass1","Planet_Mass2","Planet_Mass3"]]
      val_labels = val[["Planet_Mass1","Planet_Mass2","Planet_Mass3"]]
      test_labels = test[["Planet_Mass1","Planet_Mass2","Planet_Mass3"]]
      
      Y_train = train.pop("Planet_Count")
      Y_val = val.pop("Planet_Count")
      Y_test = test.pop("Planet_Count")      
      
#       train = train.drop(columns=["Planet_Mass1","Planet_Mass2","Planet_Mass3", "Planet_Count"])
#       val = val.drop(columns=["Planet_Mass1","Planet_Mass2","Planet_Mass3", "Planet_Count"])
#       test = test.drop(columns=["Planet_Mass1","Planet_Mass2","Planet_Mass3", "Planet_Count"])


    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    normed_train_data = norm(train)
    
    normed_val_data = norm(val)

    normed_test_data = norm(test)
    # print(normed_train_data)
    print("[INFO] Done...")

    return normed_train_data, normed_test_data, normed_val_data, train_labels, test_labels, val_labels, Y_train, Y_val, Y_test

def custom_augmentation(np_tensor, X_res, Y_res):
    
    '''
    This function is called to crop the images when the images are loaded 
    using the ImageDataGenerator Keras function. This custom augmentation function only works for
    3 res as given below. For other resolution the image needs to the cropped appropiately.
    
    '''
    
    # # # dimensions for cropping the image
    if X_res == 128:
    
      top = 80
      left = 130
      bottom = 410
      right = 440
      image = np.squeeze(np_tensor) 
      crop_image = image[top:bottom, left:right]
      crop_image = cv2.resize(crop_image, (X_res, Y_res)) 
      crop_image = k.preprocessing.image.img_to_array(crop_image)[:,:,::-1]/255.
      datagen = k.preprocessing.image.ImageDataGenerator(featurewise_center=True, zca_whitening=True)
      crop_image = datagen.standardize(crop_image)
    
    if X_res == 256:
    
      top = 80
      left = 130
      bottom = 410
      right = 440
      image = np.squeeze(np_tensor) 
      crop_image = image[top:bottom, left:right]
      crop_image = cv2.resize(crop_image, (X_res, Y_res)) 
      crop_image = k.preprocessing.image.img_to_array(crop_image)[:,:,::-1]/255.  # [:,:,::-1] converts opencv BRG convention to RBG
      datagen = k.preprocessing.image.ImageDataGenerator(featurewise_center=True, zca_whitening=True)
      crop_image = datagen.standardize(crop_image)
    
    if X_res == 512:

      top = 80
      left = 130
      bottom = 410
      right = 440
      image = np.squeeze(np_tensor) 
      crop_image = image[top:bottom, left:right]
      crop_image = cv2.resize(crop_image, (X_res, Y_res)) 
      crop_image = k.preprocessing.image.img_to_array(crop_image)[:,:,::-1]/255. 
      datagen = k.preprocessing.image.ImageDataGenerator(featurewise_center=True, zca_whitening=True)
      crop_image = datagen.standardize(crop_image)

    return crop_image

def load_disk_images(dataset, X_res, Y_res, Type):

    ''' Input : dataset with path to the images 
        Output : Images set for either test or train
    '''

    print("[INFO] Loading images from {} data..".format(Type))
    images = []
    for image_path in dataset["image_path"]:
         
        ## read the image corresponding to the path
        try:
            imagePath = image_path ## for regular code 
            image = cv2.imread(imagePath)
            
        except TypeError:
            imagePath = '..'+image_path[33:] ## when reading path from the ones gnerated in COLAB as the address in COLAB gets modified
            image = cv2.imread(imagePath)
            
        
        crop_image = custom_augmentation(image, X_res, Y_res)
        # print("working")

        ## hist Normalization not used yet
        # crop_image = exposure.equalize_hist(crop_image)
        # crop_image = exposure.equalize_adapthist(crop_image, clip_limit=0.001)


        images.append(crop_image)
    print("{} Images are loaded".format(Type))
    return np.array(images)






###########






def load_csv(folder_address, filtering=None,drop=None):

    '''
    Input : Address to the data folder
    Filtering : To select the data
    Drop: If true it drops all feature except Aspect ratio
    Output : Return a csv with parameter data and path to the image

    '''

    dataset0 = pd.concat(map(pd.read_csv, glob.glob(folder_address + '*.csv')), ignore_index=True)

    ################## DATA Filtering #############
    if filtering is True:

        # Filtering 1
        dataset0 = dataset0[dataset0['Dust_gap_1'] > 0.05]  # filtering out very narrow gaps
        dataset = pd.concat([dataset0], ignore_index=True).sort_index()  # important when merging multiple datasets
        # df = shuffle(dataset)
        # dataset = df.reset_index(drop=True)
        dataset['Planet_Mass'] = dataset['Planet_Mass'] / (3 * 10**-6)  # writing in unit of earth mass

        # Filtering 2 (removing simualation with more than two gaps)
        dataset = dataset[dataset['#_DG'] < 2]  # keeping one and two dust gap disks
        # dataset_filtered = dataset.drop(columns=['Sample#']) # dropping the Sample#

        # dataset_filtered.to_csv('data_folder/dataset_filered.csv')   # saving the filtered data as csv file for future reference
        dataset = dataset.drop(columns=['Gas_gap_1', 'Dust_depth_1', 'Dust_gap_1', 'Dust_gap_2', 'Dust_depth_2', 'Gas_depth_1', '#_DG', '#_GG'])  # droping the unimportant columns
        # dataset.to_csv('../data_folder/dataset_filered.csv')
        # dataset = dataset[['Sample#','Planet_Mass']] # droping the unimportant columns
    #     dataset = dataset.sort_values(by="Sample#")

        if drop != None: ## added to just keep the aspect ratio
            dataset = dataset.drop(columns=['Epsilon','Alpha','Stokes','SigmaSlope'])
        
        ## cleaning the data##
        dataset.isna().sum()  # summing the number of na
        dataset0 = dataset.dropna()

    return dataset0


def parse_time_series_data(folder_addrss, list_of_orbits,path, data_filters=True,drop=None):

    ''' Input : Address to the data folder and ahte orbits
        Drop: If true it drops all feature except Aspect ratio

        Output : Return a conacted csv with parameter data and path to the image with parse_dataset function
           
    '''
    dataset_complete = []
    # appended_data = []
    print("[INFO] preparing the dataframe from differnt times...")
    for i in range(len(list_of_orbits)):

        folder_address = folder_addrss + list_of_orbits[i] + '/'

        print("Reading the image paths and data from folder:", folder_address)
        # Loading the dataset for a given orbit
        dataset = parse_dataset(folder_address,filtering=data_filters,drop=drop)

        # Appending the data from the pandas dataframe for each orbits
        dataset_complete.append(dataset)
    print("[INFO] The concatination of dataframes from differnt times are now complete")
    dataset_complete = pd.concat(dataset_complete, ignore_index=True, axis=0)
    dataset_complete.to_csv(path+'data_folder/dataset_complete.csv')  # saving as dataset_complete.csv
    return dataset_complete




def process_data_for_test(test, path):

    ''' 
    ### ADDED ON 7 JANUARY TO TEST DPCNET ON HIGH MASS DATA####

        Input : CSV containing data and path to the image
        path : to get the stat files from the original trained data
        Output : Return normalized data (z normalization is used)
    '''

    print("[INFO] preparing the normalized data TEST...")
    try: 
        # train  = train.drop(columns=['Sample#', 'file']) ## dropping the necessary files
        test   = test.drop(columns=['Sample#', 'file']) ## dropping the necessary files
    except KeyError :
        pass
    
    

    train_stats = pd.read_csv(path+'data_folder/train_stats.csv',index_col=0)
    # print(train_stats)
    ## The labels are not normalized
    
    test_labels = test.pop("Planet_Mass")


    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    

    normed_test_data = norm(test)
#     print(normed_train_data)
    print("[INFO] Done...")
    return normed_test_data, test_labels






def plot_history(history, path, Model,Network= None,res =None):
    try:
        hist = pd.DataFrame(history.history) ## is the data asalready a dataframe no need to convert
        path1 = path+'figures'
    except AttributeError:
        hist = history
        path1 = path+'figures_paper'
    hist['epoch'] = hist.index
    print(path1)
    plt.figure(figsize=(5, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error ($M_\oplus$)')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train ')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 100])
    plt.legend()
    
    if Model == 'CNN':
        plt.title("Single-input DPCNet")
        if Network == None:
           plt.savefig(path1+'/MAEvalidation_loss_CNN.pdf', format='pdf', dpi=300)
        else:
           plt.savefig(path1+'/MAEvalidation_loss_{}_{}.pdf'.format(Network,str(res)), format='pdf', dpi=300)
    else:
        plt.title("Multi-input DPCNet")
        if Network == None:
           plt.savefig(path1+'/MAEvalidation_loss_hybrid.pdf', format='pdf', dpi=300)
        else:
           plt.savefig(path1+'/MAEvalidation_loss_{}_{}_hybrid.pdf'.format(Network,str(res)), format='pdf', dpi=300)
       # plt.savefig(path1+'/MAEvalidation_loss_Hybrid.pdf', format='pdf', dpi=300)

    plt.figure(figsize=(5, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ($M_\oplus^2$)')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                 label='Validation ')
    plt.ylim([0, 3000])
    #   plt.xlim([0,700])
    #   plt.yscale("log")
    plt.legend()
    plt.tick_params(labelsize=10)
    plt.tick_params(axis='both', which='major', length=6, width=2)
    plt.tick_params(axis='both', which='minor', length=3, width=1.3)
    plt.tight_layout()
    if Model == 'CNN':
        plt.title("Single-input DPCNet")
        if Network == None:
           plt.savefig(path1+'/MSEvalidation_loss_CNN.pdf', format='pdf', dpi=300)
        else:
           plt.savefig(path1+'/MSEvalidation_loss_{}_{}.pdf'.format(Network,str(res)), format='pdf', dpi=300)
       # plt.savefig(path1+'/MSEvalidation_loss_CNN.pdf', format='pdf', dpi=300)     
    else:
        plt.title("Multi-input DPCNet")
        if Network == None:
           plt.savefig(path1+'/MSEvalidation_loss_hybrid.pdf', format='pdf', dpi=300)
        else:
           plt.savefig(path1+'/MSEvalidation_loss_{}_{}_hybrid.pdf'.format(Network,str(res)), format='pdf', dpi=300)
        #plt.savefig(path1+'/MSEvalidation_loss_Hybrid.pdf', format='pdf', dpi=300)

    plt.show()
