# Read files in /errors/ and extract data from them
import os
import re
import pandas as pd
import numpy as np





def extract_data(file_name):
    """
    Extract data from a file
    """
    output = []
    
    brightness_values = {'0.0':0.0,'0.05':0.05,'0.1':0.1,'0.15':0.15,'0.2':0.2,'0.25':0.25,'0.3':0.3,'0.35':0.35,'0.4':0.4,'0.45':0.45,'0.5':0.5,
                                    '0.55':0.55,'0.6':0.6,'0.65':0.65,'0.7':0.7,'0.75':0.75,'0.8':0.8,'0.85':0.85,'0.9':0.9,'0.95':0.95,'1.0':1.0}
    terms_to_extract = ['0_class_acc', '1_class_acc', '2_class_acc', '3_class_acc', '4_class_acc', '5_class_acc', '6_class_acc', '7_class_acc', '8_class_acc', '9_class_acc','val_acc1']
    labels = {'0_class_acc':"airplane", '1_class_acc':"automobile", '2_class_acc':"bird", '3_class_acc':"cat", '4_class_acc':"deer", '5_class_acc':"dog", 
        '6_class_acc':"frog", '7_class_acc':"horse", '8_class_acc':"ship", '9_class_acc':"truck",'val_acc1':"val_acc1"}

    #terms_to_extract = [str(x) + '_class_acc' for x in range(0,100)]
    #terms_to_extract.append('val_acc1')
    short_file_name = file_name.split('/')[-1]
    model = short_file_name.split('_')[0]
    brightness = short_file_name.split('_')[-2]
    seed = int(short_file_name.split('_')[2])
    print(brightness)
    if brightness in brightness_values.keys():
        brightness = brightness_values[brightness]
    else:
        return None
    
    
    #print("we here")



    # Open file
    with open(file_name, 'r') as f:
        # Read file
        data = f.read()
        
        # Extract data
        data = data.split("Run summary:")[-1].split("\n")
        # keep only the lines that contain the terms to extract in them
        data = [line for line in data if any(term in line for term in terms_to_extract)]
        output_line = {'model':model.split(".")[0], 'contrast':brightness,"seed":seed}
        for line in data :
            value = float(line.split(' ')[-1])
            key = line.split(' ')[-2]
            output_line[labels[key]] = value
        output.append(output_line)
        # Return data
        
        return output


if __name__   == "__main__":
    # Get all files in the folder
    files = os.listdir('../errors/')
    # Extract data from each file
    data = []
    for file in files:
        
        if file.endswith('.out'):
            
            try:

                fa = extract_data('../errors/' + file)
                #print(fa)

                if fa != None:
                    #print(fa)
                    data += fa
                else :
                    continue
            except :
                continue #print(file)
    # Convert to pandas dataframe
    data = pd.DataFrame(data)

    # group by model and brightness and compute mean of all values
    data_mean = data.groupby(['model','contrast']).mean().reset_index()
    # group by model and brightness and compute mean of all values
    data_std = data.groupby(['model','contrast']).std().reset_index()
    # Save to csv
    data_mean.to_csv('data_mean.csv', index=False)
    data_std.to_csv('data_std.csv', index=False)

    # Save data
    data.to_csv('./contrast_cifar10.csv', index=False)