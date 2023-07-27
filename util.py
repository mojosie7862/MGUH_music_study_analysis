
def get_markers(vmrk_file):
    file = open(vmrk_file)
    marker_data = []
    for line in file.readlines():
        if line.startswith('Mk'):
            marker = line.split(',')
            marker_data.append(marker)
    for i, marker_ls in enumerate(marker_data):
        if marker_data[i+1][1] == 'b1_stop_music_start':
            b1_start = int(marker_ls[2])
            music_start = int(marker_data[i+1][2])
            music_stop = int(marker_data[i+2][2])
            b2_start = int(marker_data[i+3][2])
            break
    return b1_start, music_start, music_stop, b2_start


def subset_intervals(raw_data, b1_start, music_start, music_stop, b2_stop):
    # take section of music interval to get same time resolution
    music_int_stop = music_stop-7500
    music_int_start = music_int_stop-150000
    b1_start = music_start - 150000
    b2_stop = music_stop + 150000
    b1_raw = raw_data.iloc[b1_start:music_start, :].reset_index()
    music_raw = raw_data.iloc[music_int_start:music_int_stop, :].reset_index()
    b2_raw = raw_data.iloc[music_stop:b2_stop, :].reset_index()
    return b1_raw, music_raw, b2_raw


def print_dict(dict1, dict2):
    for d1, d2 in zip(dict1.items(), dict2.items()):
        print(d1, d2)


def add_int_prefix(col_name, interval):
    new_col_name = interval+'_'+col_name
    return new_col_name

# modifying hdr file for NME
def clean_vhdr_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
        data = data.replace('%', '%%')
        data = data.replace('Phys. chn. Resolution / unit', 'Phys. chn.     Resolution / Unit')
    with open(filename, 'w') as file:
        file.write(data)