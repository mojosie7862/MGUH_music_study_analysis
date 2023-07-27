import os
import shutil
import subprocess

# if file starts with NHC - move to Raw_NHC_AIM
# if file starts with ICM - move to Raw_ICM_AIM
# make copy of said NHC file and paste it to the Copy_NHC_AIM folder
# make copy of said ICM file and paste it to the Copy_ICM_AIM folder

aim_output_dir = r"C:/Users/kanwali/Box/MICU_AIM_data/Raw_Files"
raw_nhc_dir = r"C:/Users/kanwali/Box/MICU_AIM_data/Raw_NHC_AIM"
raw_icm_dir = r"C:/Users/kanwali/Box/MICU_AIM_data/Raw_ICM_AIM"
copy_nhc_dir = r"C:/Users/kanwali/Box/MICU_AIM_data/Copy_NHC_AIM"
copy_icm_dir = r"C:/Users/kanwali/Box/MICU_AIM_data/Copy_ICM_AIM"

id_set = set()
for fn in os.listdir(aim_output_dir):
    original_path = aim_output_dir + "/" + fn
    if fn.startswith('NHC'):
        target_raw_path = raw_nhc_dir + "/" + fn
        target_copy_path = copy_nhc_dir + "/" + fn
        id_set.add(fn[:9])
    elif fn.startswith('ICM'):
        target_raw_path = raw_icm_dir + "/" + fn
        target_copy_path = copy_icm_dir + "/" + fn
        id_set.add(fn[:13])
    else:
        continue   # don't over-write previous file if there's a test file in there

    shutil.move(original_path, target_raw_path)
    print("moved", fn, "to", target_raw_path)
    shutil.copy(target_raw_path, target_copy_path)
    print("copied", fn, "to", target_copy_path)

# run clean_vhdr_files.py - on both Copy_NHC_AIM and Copy_ICM_AIM folders

subprocess.run(['python', 'clean_vhdr_files.py', 'nhc'])
subprocess.run(['python', 'clean_vhdr_files.py', 'icm'])

print("check the markers in the following copied files:")
for i in id_set:
    print(i+'.vmrk')

# check .vmrk files for 4 markers and if there are less or more print name of id to terminal and move on to next ID

#move ids with clean vhdr and vmrk files to the clean folder

#run process_aim_tocsv.py
