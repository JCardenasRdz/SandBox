from .package_control import *
# modules
import nibabel as nib
import os

def get_patient_list(path_):
    df = pd.read_csv(path_)
    patient_list_ = df[df.columns[0]].values
    return patient_list_

def load_data(DirPath,patientID, element = 'flair'):
    if element == 'flair':
        experiment = 0

    elif element == 'seg':
        experiment = 1

    elif element == 't1':
        experiment = 2

    elif element == 't1ce':
        experiment = 3

    elif element == 't1ce':
        experiment = 4

    path_to_patient, file_list = get_path_and_files(DirPath, patientID)
    full_path = os.path.join(path_to_patient, file_list[  experiment ])
    I = nib.load(full_path).get_data()

    return I

def get_path_and_files(DirPath_, patientID_):
    for dirName, subdirList, fileList in os.walk(DirPath_):
        if dirName.split('/')[-1] == patientID_:
            path_ = dirName
            file_list = fileList
            #print(path_)

    return path_, file_list


def count_classes(DirPath,patientID):
    # load and flatten data
    #I = load_data(DirPath, patientID, element = 'seg').flatten()
    #df = pd.DataFrame(I, columns=['Class'])
    #df[patientID] = 1
    #df = df.groupby(['Class']).count()

    I = load_data(DirPath,patientID, element = 'seg').flatten()
    df = pd.DataFrame(I, columns=[patientID])
    df = df[patientID].value_counts()


    return df
