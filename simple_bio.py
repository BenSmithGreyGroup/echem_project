def simple_bio(file, directory, save=True, mass=None, cap_label='Capacity/mA.h', volt_label='Ewe/V'):
    """
    Simply loading in biologic data into pandas. Assumes the mass is in g and the columns are labelled in the default format
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
                mass = float(mass_line.split()[-2])
    searchfile.close()
    if mass is None:
        print(f'{file} - WARNING: no mass supplied or found in file - no specific capacity information will be processed \n supply mass as variable in function call')
    # -1 because want column headings and that's the number that gets them
    header_lines_to_skip = int(header_lines_to_skip_line.split()[-1]) - 1

    # setting up the dataframe and getting the column headings
    df = pd.read_csv(directory + file, sep='\t', header=header_lines_to_skip,
                       index_col=False, skip_blank_lines=False, encoding="latin-1")
    df['Specific Capacity / mAh/g'] = df[cap_label]/mass
    return df