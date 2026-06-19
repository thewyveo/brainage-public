# single mri
"""
synthba -i ./data/raw/BraTS/BraTS-GLI-00005-100-t1n.nii.gz -o ./data/predictions --device cpu
"""

# batch mri
#synthba -i ./data/raw/BraTS/ -o ./data/predictions --device cpu --batch_size 4
"""
py -3.10 "C:\Users\P102179\AppData\Local\Programs\Python\Python310\Scripts\synthba.exe" -i ./data/raw/BraTS/ -o ./data/predictions --device cpu --batch_size 4
py -3.10 "C:\Users\P102179\AppData\Local\Programs\Python\Python310\Scripts\synthba.exe" -i ./data/exp0_gligan_brats24_ixi_t1_only1tumor_flat/ -o ./data/predictions --device cpu --batch_size 4
py -3.10 "C:\Users\P102179\AppData\Local\Programs\Python\Python310\Scripts\synthba.exe" -i ./data/ixi_t1_only/ -o ./data/predictions --device cpu --batch_size 4 --skip-prep
"""

# csv
"""
synthba -i ./data/labels/BraTS_24.xlsx -o ./data/predictions --device cpu --batch_size 4
"""
#! REQUIRES PATH