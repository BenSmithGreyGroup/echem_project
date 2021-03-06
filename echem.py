from galvani import MPRfile
from galvani import res2sqlite as r2s
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import splev, splrep
import sqlite3
from bokeh.plotting import figure
import os
import matplotlib.pyplot as plt

# Different cyclers name their columns slightly differently 
# These dictionaries are guides for the main things you want to plot

res_col_dict = {'Voltage': 'Voltage',
                'Capacity': 'Capacity'}

mpr_col_dict = {'Voltage': 'Ewe/V',
                'Capacity': 'Capacity'}


def echem_file_loader(filepath):
    extension = os.path.splitext(filepath)[-1].lower()
    # Biologic file
    if extension == '.mpr':
        gal_file = MPRfile(os.path.join(filepath))
        df = pd.DataFrame(data=gal_file.data)
        df = biologic_processing(df)

    #arbin .res file - uses an sql server
    elif extension == '.res': 
        Output_file = 'placeholder_string'
        r2s.convert_arbin_to_sqlite(os.path.join(filepath), Output_file)
        dat = sqlite3.connect(Output_file)
        query = dat.execute("SELECT * From Channel_Normal_Table")
        cols = [column[0] for column in query.description]
        df = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
        dat.close()
        df.set_index('Data_Point', inplace=True)
        df.sort_index(inplace=True)
        df['Capacity'] = df['Charge_Capacity'] + df['Discharge_Capacity']

    elif extension == '.txt':
        df = pd.read_csv(os.path.join(filepath), sep='\t')
        # Checking columns are ex exact match
        if set(['time /s', 'I /mA', 'E /V']) - set(df.columns) == set([]):
            df = ivium_processing(df)
        else:
            print('Columns did not match known .txt column layous')
    
    elif extension in ['.xlsx', '.xls']:
        xlsx = pd.ExcelFile(os.path.join(filepath))
        names = xlsx.sheet_names
        # Edit this in a bit
        if len(names) == 1:
            df = xlsx.parse(0)
            df = new_land_processing(df)
        elif "Record" in names[0]:
            df_list = [xlsx.parse(0)]
            col_names = df_list[0].columns

            for sheet_name in names[1:]:
                if "Record" in sheet_name:
                    if len(xlsx.parse(sheet_name, header=None)) != 0:
                        df_list.append(xlsx.parse(sheet_name, header=None))
            for sheet in data_sheets:
                sheet.columns = col_names
            df = pd.concat(df_list)
            df.set_index('Index', inplace=True)
            df = old_land_processing(df)
        else:
            df_list = []
            for count, name in enumerate(names):
                if 'Channel' in name and 'Chart' not in name:
                    df_list.append(xlsx.parse(count))
            if len(df_list) > 0:
                df = pd.concat(df_list)
                df = arbin_excel(df)
            else:
                raise SheetNameError('Names of sheets not recognised')
    else:
        print(extension)
        print('We got to here')
    return df

def biologic_processing(df):
    # Dealing with the different column layouts for biologic files
    # Renames Ewe/V to Voltage and the capacity column to Capacity
    if 'half cycle' in df.columns:
        if df['half cycle'].min() == 0:
            df['half cycle'] = df['half cycle'] + 1
    if ('Q charge/discharge/mA.h' in df.columns) and ('half cycle') in df.columns:
        df['Capacity'] = abs(df['Q charge/discharge/mA.h'])
        df.rename(columns = {'Ewe/V':'Voltage'}, inplace = True)
        return df

    elif ('dQ/mA.h' in df.columns) and ('half cycle') in df.columns:
        df['Half cycle cap'] = abs(df['dQ/mA.h'])
        for cycle in df['half cycle'].unique():
            mask = df['half cycle'] == cycle
            cycle_idx = df.index[mask]
            df.loc[cycle_idx, 'Half cycle cap'] = df.loc[cycle_idx, 'Half cycle cap'].cumsum()
        df.rename(columns = {'Half cycle cap':'Capacity'}, inplace = True)
        df.rename(columns = {'Ewe/V':'Voltage'}, inplace = True)
        return df

    else:
        print('Unknown column layout')

def ivium_processing(df):
    df['dq'] = np.diff(df['time /s'], prepend=0)*df['I /mA']
    df['Capacity'] = df['dq'].cumsum()/3600
    def ivium_state(x):
        if x >=0:
            return 0
        else:
            return 1

    df['state'] = df['I /mA'].map(lambda x: ivium_state(x))
    df['half cycle'] = df['state'].ne(df['state'].shift()).cumsum()
    for cycle in df['half cycle'].unique():
        mask = df['half cycle'] == cycle
        idx = df.index[mask]
        df.loc[idx, 'Capacity'] = abs(df.loc[idx, 'dq']).cumsum()/3600
    df['Voltage'] = df['E /V']
    return df

def new_land_processing(df):
    # Remove half cycle == 0 for initial resting
    if 'Voltage/V' not in df.columns:
        column_to_search = df.columns[df.isin(['Index']).any()][0]
        df.columns = df[df[column_to_search] == 'Index'].iloc[0]
    df = df[df['Current/mA'].apply(type) != str]
    df = df[pd.notna(df['Current/mA'])]

    def land_state(x):
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 'R'
        else:
            print(x)
            raise ValueError('Unexpected value in current - not a number')

    df['state'] = df['Current/mA'].map(lambda x: land_state(x))

    not_rest_idx = df[df['state'] != 'R'].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()
    df['Voltage'] = df['Voltage/V']
    df['Capacity'] = df['Capacity/mAh']
    return df

def old_land_processing(df):
    df = df[df['Current/mA'].apply(type) != str]
    df = df[pd.notna(df['Current/mA'])]

    def land_state(x):
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 'R'
        else:
            print(x)
            raise ValueError('Unexpected value in current - not a number')

    df['state'] = df['Current/mA'].map(lambda x: land_state(x))
    not_rest_idx = df[df['state'] != 'R'].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()
    df['Voltage'] = df['Voltage/V']
    df['Capacity'] = df['Capacity/mAh']
    return df

def arbin_excel(df):
    df.reset_index(inplace=True)

    def arbin_state(x):
        if x > 0:
            return 0
        elif x < 0:
            return 1
        elif x == 0:
            return 'R'
        else:
            print(x)
            raise ValueError('Unexpected value in current - not a number')

    df['state'] = df['Current(A)'].map(lambda x: arbin_state(x))

    not_rest_idx = df[df['state'] != 'R'].index
    df.loc[not_rest_idx, 'cycle change'] = df.loc[not_rest_idx, 'state'].ne(df.loc[not_rest_idx, 'state'].shift())
    df['half cycle'] = (df['cycle change'] == True).cumsum()

    df['Capacity'] = df['Discharge_Capacity(Ah)'] + df['Charge_Capacity(Ah)']
    for cycle in df['half cycle'].unique():
        idx = df[df['half cycle'] == cycle].index
        df.loc[idx, 'Capacity'] = df.loc[idx, 'Capacity'] - min(df.loc[idx, 'Capacity'])
    df['Voltage'] = df['Voltage(V)']
    return df

def dqdv_single_cycle(capacity, voltage, 
                    polynomial_spline=3, s_spline=1e-5,
                    polyorder_1 = 5, window_size_1=101,
                    polyorder_2 = 5, window_size_2=1001,
                    final_smooth=True):
    
    import pandas as pd
    import numpy as np
    from scipy.interpolate import splrep, splev

    df = pd.DataFrame({'Capacity': capacity, 'Voltage':voltage})
    unique_v = df.astype(float).groupby('Voltage').mean().index
    unique_v_cap = df.astype(float).groupby('Voltage').mean()['Capacity']

    x_volt = np.linspace(min(voltage), max(voltage), num=int(1e4))
    f_lit = splrep(unique_v, unique_v_cap, k=1, s=0.0)
    y_cap = splev(x_volt, f_lit)
    smooth_cap = savgol_filter(y_cap, window_size_1, polyorder_1)

    f_smooth = splrep(x_volt, smooth_cap, k=polynomial_spline, s=s_spline)
    dqdv = splev(x_volt, f_smooth, der=1)
    smooth_dqdv = savgol_filter(dqdv, window_size_2, polyorder_2)
    if final_smooth:
        return x_volt, smooth_dqdv, y_cap
    else:
        return x_volt, dqdv, y_cap

"""
PLOTTING
"""

def charge_discharge_plot(df, full_cycle, colormap=None):
    """
    Function for plotting individual or multi but discrete charge discharge cycles
    """
    fig, ax = plt.subplots()

    try:
        iter(full_cycle)

    except TypeError:
        cycles = [full_cycle*2 -1, full_cycle*2]
        for cycle in cycles:
            mask = df['half cycle'] == cycle
            # Making sure cycle exists within the data
            if sum(mask) > 0:
                ax.plot(df['Capacity'][mask], df['Voltage'][mask])

        ax.set_xlabel('Capacity / mAh')
        ax.set_ylabel('Voltage / V')
        return fig, ax
    
    if not colormap:
        if len(full_cycle) < 11:
            colormap = 'tab10'
        elif len(full_cycle) < 21:
            colormap = 'tab20'
        else:
            raise ValueError("Too many cycles for default colormaps. Use multi_cycle_plot instead")
    
    cm = plt.get_cmap(colormap)
    for count, full_cycle_number in enumerate(full_cycle):
        cycles = [full_cycle_number*2 -1, full_cycle_number*2]
        for cycle in cycles:
            mask = df['half cycle'] == cycle
            # Making sure cycle exists within the data
            if sum(mask) > 0:
                ax.plot(df['Capacity'][mask], df['Voltage'][mask], color=cm(count))

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=cm(count), lw=2) for count, i in enumerate(full_cycle)]
    
    ax.legend(custom_lines, [f'Cycle {i}' for i in full_cycle])
    ax.set_xlabel('Capacity / mAh')
    ax.set_ylabel('Voltage / V')
    return fig, ax
    

def multi_cycle_plot(df, cycles, colormap='viridis'):
    """
    Function for plotting continuously coloured cycles (useful for large numbers)
    
    Supply the cycles as half cycle numbers e.g 1, 2 are discharge and charge for 
    first cycle
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots()
    cm = plt.get_cmap(colormap)
    norm = Normalize(vmin=int(np.ceil(min(cycles)/2)), vmax=int(np.ceil(max(cycles)/2)))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

    for cycle in cycles:
        mask = df['half cycle'] == cycle
        ax.plot(df['Capacity'][mask], df['Voltage'][mask], color=cm(norm(np.ceil(cycle/2))))

    cbar = fig.colorbar(sm)
    cbar.set_label('Cycle', rotation=270, labelpad=10)
    ax.set_ylabel('Voltage / V')
    ax.set_xlabel('Capacity / mAh')
    return fig, ax


def multi_dqdv_plot(df, cycles, colormap='viridis', 
    capacity_label='Capacity', 
    voltage_label='Voltage',
    polynomial_spline=3, s_spline=1e-5,
    polyorder_1 = 5, window_size_1=101,
    polyorder_2 = 5, window_size_2=1001,
    final_smooth=True):
    
    """
    Plotting multi dQ/dV cyles on the same plot with a colormap.
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots()
    cm = plt.get_cmap(colormap)
    norm = Normalize(vmin=int(np.ceil(min(cycles)/2)), vmax=int(np.ceil(max(cycles)/2)))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

    for cycle in cycles:
        df_cycle = df[df['half cycle'] == cycle]
        voltage, dqdv, cap = dqdv_single_cycle(df_cycle[capacity_label], 
                                    df_cycle[voltage_label], 
                                    window_size_1=window_size_1,
                                    polyorder_1=polyorder_1,
                                    s_spline=s_spline,
                                    window_size_2=window_size_2,
                                    polyorder_2=polyorder_2,
                                    final_smooth=final_smooth)
        
        ax.plot(voltage, dqdv, color=cm(norm(np.ceil(cycle/2))))

    cbar = fig.colorbar(sm)
    cbar.set_label('Cycle', rotation=270, labelpad=10)
    ax.set_xlabel('Voltage / V')
    ax.set_ylabel('dQ/dV / $mAhV^{-1}$')
    return fig, ax
