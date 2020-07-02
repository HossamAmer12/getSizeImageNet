

import os
import math
import numpy as np

# Number of samples in ImageNet validation
N = 50000

# Number of QFs
n_qfs = 10 + 1

PATH_TO_VALIDATION_IMAGES = '/home/h2amer/work/workspace/ML_TS/validation_original/'
PATH_TO_VALIDATION_IMAGES_QF = '/media/h2amer/MULTICOM-104/Customized_validation/validation_generated_QF_1/' 



# Gets the total size of the quality factor predictions
def getTotalSize(jpeg_row_id, predicted_qf):

  # create the image path
  imageID         = jpeg_row_id + 1
  original_img_ID = imageID
  folder_num      = math.ceil(original_img_ID / 1000)
  shard_num       = math.floor((original_img_ID - 1) / 10000)
  imageID         = str(imageID).zfill(8)

  if predicted_qf == 110:
    name           = 'ILSVRC2012_val_' + imageID + '.JPEG'
    path_to_image  = PATH_TO_VALIDATION_IMAGES + 'shard-' + str(shard_num) + '/' + str(folder_num) + '/' + name
  else:
    name           = 'ILSVRC2012_val_' + imageID + '-QF-' + str(predicted_qf) + '.JPEG'
    path_to_image  = PATH_TO_VALIDATION_IMAGES_QF + 'shard-' + str(shard_num) + '/' + str(folder_num) + '/' + name
  
  return os.path.getsize(path_to_image)



sizes_array = np.zeros([N, n_qfs])

for jpeg_row_id in range(N):
	for iqf, qf in enumerate(range(110, 0, -10)):
		sizes_array[jpeg_row_id][iqf] = getTotalSize(jpeg_row_id, qf)

	if  not jpeg_row_id % 10:
		print('Done %d ' % jpeg_row_id)

output_file = 'imagenet_validation_sizes_110_10.npy'
np.save(output_file, sizes_array)