import os
from util import clean_vhdr_file
import sys

group_v = sys.argv[1]
if group_v == 'nhc':
    copy_aim_directory = r"C:\Users\kanwali\Box\MICU_AIM_data\Copy_NHC_AIM"
elif group_v == 'icm':
    copy_aim_directory = r"C:\Users\kanwali\Box\MICU_AIM_data\Copy_ICM_AIM"
else:
    print('please use either "nhc" or "icm" argument at the end of the command')
    exit()


print("cleaning", group_v, ".vhdr files")

for file in os.listdir(copy_aim_directory):
    filename = copy_aim_directory+'/'+file
    f_suffix = filename[-5:]
    if f_suffix == ".vhdr":
        clean_vhdr_file(filename)

print("Done.")



