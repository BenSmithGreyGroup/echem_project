import pandas as pd
import numpy as np

def process_bio(directory, file, save=-False, savedirectory=None, remove_short_cycles=True, remove_cv=True,
                    file_check_dict=None, mass=None):

    """

    Function to read and process biologic EC data files exported from EC-lab, returns a pandas dataframe of
    the relevant data. Can also save this dataframe as a csv file in the directory of your choice. Additionally certain
    filters can be toggled and a dictionary can be supplied to record anomalous cycles that may require more close
    analysis.

    Note if you don't supply a mass, the function will look for a line containing "Characteristic mass" and take the
    number from there. This is currently assuming this is in mg (all files I have seen this is the unit) If you store
    the mass as a different unit you will need to edit the scaling factor (1e-3) currently to whatever is suitable to
    convert from your unit to g

    :param savedirectory: Directory to save .csv file to if desired
    :param file: .mpt or .txt file to read from EC lab
    :param save: If .csv file is to be saved - True, else False, default False
    :param directory: Directory to read .mpt or .txt file from - end in '/'
    :param remove_short_cycles: If you want to delete cycles < 10s - not really a full cycle but sometimes appear
                                default = True
    :param remove_cv: Remove constant voltage cycles - default =True
    :param file_check_dict: If you want to have a check dictionary -
                            to save any weird results that might need more checking
    :param mass If you want to input a manual mass to overide what is in the file header - units = grams
    :return:
    """
    searchfile = open(directory + file, "r", encoding="latin-1", errors="ignore")

    header_lines_to_skip_line = 'Na'

    # finding number of header lines to skip and characteristic mass for later
    for line in searchfile:
        if "header lines" in line:
            header_lines_to_skip_line = line
        if mass is None:
            if "Characteristic mass" in line:
                mass_line = line
                mass = float(mass_line.split()[-2]) * 1e-3
    searchfile.close()
    if mass is None:
        print('WARNING: no mass supplied or found in file - no specific capacity information will be processed \n supply mass as variable in function call')
    # -1 because want column headings and that's the number that gets them
    header_lines_to_skip = int(header_lines_to_skip_line.split()[-1]) - 1

    # setting up the dataframe and getting the column headings
    df = pd.read_csv(directory + file, sep='\t', header=header_lines_to_skip,
                       index_col=False, skip_blank_lines=False, encoding="latin-1")

    # Removing rest mode
    df = df[df['mode'] != 3]

    # Getting cyclenumber - this is in half cycles
    df['Cycle'] = df['ox/red'].ne(df['ox/red'].shift()).cumsum()
    # removing cycles less than 10s - conditional
    if remove_short_cycles:
        for cycle in df['Cycle'].unique():
            if 'cycle time/s' in df.columns:
                if df[df['Cycle'] == cycle]['cycle time/s'].max() < 10:
                    print(file)
                    print('Cycle {}, cycle time {}s'.format(cycle, df[df['Cycle'] == cycle]['cycle time/s'].max()))
                    df = df[df['Cycle'] != cycle]
            else:
                print('No cycle time/s column exported')

    # Conditionally removing constant voltage cycles
    if remove_cv:
        df = df[df['mode'] != 2]

    df['Cycle'] = df['ox/red'].ne(df['ox/red'].shift()).cumsum()

    if file_check_dict is not None:
        for cyclenum, cycledata in df.groupby('Cycle'):
            # print('Cycle {} length {}'.format(cyclenum, len(cycledata['Cycle'])))
            if len(cycledata['Cycle']) < 100:
                file_check_dict.setdefault(file, [[], []])
                file_check_dict[file][0].append(cyclenum)
        dv_dict = {cyclenum:
                       np.array(df[df['Cycle'] == cyclenum]['Ewe/V'])[-1]
                       - np.array(df[df['Cycle'] == cyclenum]['Ewe/V'])[0]
                   for cyclenum in df['Cycle'].unique()}
        for cycle in list(range(2, df['Cycle'].max() + 1, 2)):
            if dv_dict[cycle] < 0:
                file_check_dict.setdefault(file, [[], []])
                file_check_dict[file][1].append(cycle)
        for cycle in list(range(1, df['Cycle'].max() + 1, 2)):
            if dv_dict[cycle] > 0:
                file_check_dict.setdefault(file, [[], []])
                file_check_dict[file][1].append(cycle)

    # Export dataframe - possibly better way of doing this but oh well
    dataframe = pd.DataFrame()
    dataframe['Cycle'] = df['Cycle']
    # For own personal use this was to link together samples with many files - can be done later though
    # dataframe['Filenumber'] = bio_file_df.reset_index().set_index('File').loc[file, 'Filenumber']

    dataframe['Machine'] = 'bio'
    dataframe['Mode'] = df['mode']
    dataframe['Current/mA'] = df['<I>/mA']

    # Couple of conditional things on mass - written out like this so the columns are in the "right" order -
    # could be shortened I'm sure
    if mass is not None:
        dataframe['Mass'] = mass

    dataframe['Voltage/V'] = df['Ewe/V']
    dataframe['Capacity/mAh'] = abs(df.groupby(['Cycle'])['dq/mA.h'].cumsum())

    if mass is not None:
        dataframe['SCapacity/mAh/g'] = dataframe['Capacity/mAh'] / mass

    if 'd(Q-Qo)/dE/mA.h/V' in df.columns:
        dataframe['d(Q-Qo)/dE/mA.h/V'] = df['d(Q-Qo)/dE/mA.h/V']
        if mass is not None:
            dataframe['dQdV specific'] = df['d(Q-Qo)/dE/mA.h/V'] / mass

    if save:
        filename = file.split('.')[0]
        dataframe.to_csv(savedirectory + '{}.csv'.format(filename))
    return dataframe

if __name__ == "__main__":
    test_directory = '/Users/ben/Google Drive/Research Project/Colab Data/Final Bio data/'
    test_file = 'yj277_NP59-SiNP_CMC cycle test_C12_InQ.mpt'
    test_save_dir = '/Users/ben/Desktop/'
    check_dict = {}

    df = process_bio(test_directory, test_file, save=True, savedirectory=test_save_dir, file_check_dict=check_dict, mass=None)
    print(df.head())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,10))
    for cycle in df['Cycle'].unique():
        plt.plot(df[df['Cycle'] == cycle]['SCapacity/mAh/g'], df[df['Cycle'] == cycle]['Voltage/V'])
    plt.savefig(f"{test_save_dir}test_fig")
    print(check_dict)