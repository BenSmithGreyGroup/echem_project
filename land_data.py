import pandas as pd
import numpy as np
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt

def process_land(directory, file, save=False, savedirectory=None, remove_short_cycles=True, remove_cv=True,
                 file_check_dict=None, mass=None, plot=True):
    """

    Function to read and process land EC data files exported landdt, returns a pandas dataframe of
    the relevant data. Can also save this dataframe as a csv file in the directory of your choice. Additionally certain
    filters can be toggled and a dictionary can be supplied to record anomalous cycles that may require more close
    analysis.

    Note if you don't supply a mass, the function will look for a column called SCapacity/mAh/g in the export
    - if neither are here it will not give specific capacity information

    :param savedirectory: Directory to save .csv file to if desired
    :param file: .xlsx or .xlx file to read from Landdt
    :param save: If .csv file is to be saved - True, else False, default False
    :param directory: Directory to read .mpt or .txt file from - end in '/'
    :param remove_short_cycles: If you want to delete cycles < 10 data points - not really a full cycle but sometimes appear
                                default = True
    :param remove_cv: Remove constant voltage cycles - default =True
    :param file_check_dict: If you want to have a check dictionary -
                            to save any weird results that might need more checking
    :param mass If you want to input a manual mass to overide what is in the file header - units = grams
    :return:
    """
    # Loading in the file and stitching together the multiple excel sheets if there are any
    xlsx = pd.ExcelFile(directory + file)
    names = xlsx.sheet_names

    data_sheets = [xlsx.parse(0)]
    # Finding all sheets with Record in after the first
    for sheet_name in names[1:]:
        if "Record" in sheet_name:
            if len(xlsx.parse(sheet_name, header=None)) != 0:
                data_sheets.append(xlsx.parse(sheet_name, header=None))

    # fixing columns for subsequent data sheets (no header for these in input file
    for sheet in data_sheets:
        sheet.columns = data_sheets[0].columns
    df = pd.concat(data_sheets)
    df.set_index('Index', inplace=True)

    # Removing any mode rest data so it doesn't come up as the first cycle
    df = df[df['Current/mA'] != 0.0]

    # If the state is not exported try to infer from sign of current
    def state_func(x):
        if x > 0:
            return 'Charge'
        elif x < 0:
            return 'Discharge'
        else:
            raise ValueError('Current = 0.0 - cannot decide if charge or discharge')

    if 'State' not in [i for i in df.columns]:
        df['State'] = df['Current/mA'].apply(state_func)

    df['Cycle'] = df['State'].ne(df['State'].shift()).cumsum()
    # Optionally removing small cycles - won't count them for cycle num etc - removes some anomalous cycles
    if remove_short_cycles:
        for cycle in df['Cycle'].unique():
            if len(df[df['Cycle'] == cycle]) < 10:
                df = df[df['Cycle'] != cycle]

    df['Cycle'] = df['State'].ne(df['State'].shift()).cumsum()

    # Optionally removing CV - won't recount cycles

    if remove_cv:
        df = df[df['State'] != 'D_CV']
        df = df[df['State'] != 'C_CV']

    # if len(df[df['State'] == 'C_CV']) > 5:
    #     df = df
    # else:
    #     df = df[df['State'] != 'D_CV']
    #     df = df[df['State'] != 'C_CV']
    #     df['Cycle'] = df['State'].ne(df['State'].shift()).cumsum()

    # Escape if the filters remove every cycle - i.e if all are rest or CV and these are specified to be removed
    if len(df) == 0:
        print(f'{file} no cycles remaining after filters')
        return

    # Optional file check functionality - will return the supplied dictionary with the cycles that may be anomalous
    if file_check_dict is not None:
        for cyclenum, cycledata in df.groupby('Cycle'):
            # print('Cycle {} length {}'.format(cyclenum, len(cycledata['Cycle'])))
            if len(cycledata['Cycle']) < 100:
                file_check_dict.setdefault(file, [[], []])
                file_check_dict[file][0].append(cyclenum)
        dv_dict = {cyclenum:
                       np.array(df[df['Cycle'] == cyclenum]['Voltage/V'])[-1]
                       - np.array(df[df['Cycle'] == cyclenum]['Voltage/V'])[0]
                   for cyclenum in df['Cycle'].unique()}

        for cycle in list(range(2, int(df['Cycle'].max() + 1), 2)):
            if dv_dict[cycle] < 0:
                file_check_dict.setdefault(file, [[], []])
                file_check_dict[file][1].append(cycle)
        for cycle in list(range(1, int(df['Cycle'].max() + 1), 2)):
            if dv_dict[cycle] > 0:
                file_check_dict.setdefault(file, [[], []])
                file_check_dict[file][1].append(cycle)

    # Setting up and populating the export dataframe - will have same columns as the bio export function
    dataframe = pd.DataFrame()
    dataframe['Cycle'] = df['Cycle']
    dataframe['Machine'] = 'land'
    dataframe['Mode'] = df['State']
    dataframe['Current/mA'] = df['Current/mA']
    dataframe['Voltage/V'] = df['Voltage/V']
    dataframe['Capacity/mAh'] = df['Capacity/mAh']

    if mass is not None:
        dataframe['SCapacity/mAh/g'] = dataframe['Capacity/mAh'] / mass

    elif 'SCapacity/mAh/g' in [i for i in df.columns]:
        dataframe['SCapacity/mAh/g'] = df['SCapacity/mAh/g']
    else:
        print(f'{file} - WARNING, no mass supplied and no SCapacity exported - no SCapacity will be processed')

    filename = file.split('.')[0]

    if save:
        dataframe.to_csv(savedirectory + '{}.csv'.format(filename))

    if plot:
        for cycle in df['Cycle'].unique():
            plt.plot(df[df['Cycle'] == cycle]['Capacity/mAh'], df[df['Cycle'] == cycle]['Voltage/V'])
        plt.xlabel('Capacity/mAh')
        plt.ylabel('Voltage/V')
        plt.show()
        plt.close()
    return dataframe

if __name__ == "__main__":
    test_directory = '/Users/ben/Google Drive/Research Project/Colab Data/land data yanting correct/'
    test_file = 'SS219_FEC_100170810_015_7.xls'
    test_save_dir = '/Users/ben/Desktop/'
    check_dict = {}

    df = process_land(test_directory, test_file, save=True, savedirectory=test_save_dir, file_check_dict=check_dict,
                     mass=None, plot=False)
    print(df.head())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for cycle in df['Cycle'].unique():
        plt.plot(df[df['Cycle'] == cycle]['SCapacity/mAh/g'], df[df['Cycle'] == cycle]['Voltage/V'])
    plt.savefig(f"{test_save_dir}test_fig")
    print(check_dict)