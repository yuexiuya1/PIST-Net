from astropy.io import fits
import numpy as np
import os
source_folder="./data/pic96/"
destination_folders="./data/pic"
orginalSize=96
targetSize=[16,24,32,40,48,56,64,72,80,88]
for tSize in targetSize:
    destfolder=destination_folders+str(tSize)+"/"
    os.makedirs(destfolder, exist_ok=True)
for filename in os.listdir(source_folder):
    if filename.endswith('.fits'):
        # 构建完整文件路径
        file_path = os.path.join(source_folder, filename)
        hdu=fits.open(file_path)
        data =  hdu[0].data.astype(np.float32)
        for tSize in targetSize:
            destfolder=destination_folders+str(tSize)+"/"
            new_file_path = os.path.join(destfolder, filename)

            cut=(orginalSize-tSize)//2
            processed_data = data[:,cut:orginalSize-cut, cut:orginalSize-cut]
            hdu_cropped = fits.PrimaryHDU(data=processed_data)
            hdu_cropped.writeto(new_file_path, overwrite=True)
        hdu.close()

