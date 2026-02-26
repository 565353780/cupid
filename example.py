import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['SPCONV_ALGO'] = 'native'  # Use 'auto' for faster repeated runs

import imageio
import numpy as np
from PIL import Image
from cupid.pipelines import Cupid3DPipeline
from cupid.utils import render_utils, sample_utils
from cupid.utils.align_utils import save_mesh


home = os.environ['HOME']

model_folder_path = home + '/chLi/Model/CUPID/Cupid/'

data_folder_path = home + '/chLi/Dataset/GS/'

image_file_path = data_folder_path + 'woman3.png'
save_result_folder_path = data_folder_path + 'cupid/woman3/'
os.makedirs(save_result_folder_path, exist_ok=True)

# Load pipeline
pipeline = Cupid3DPipeline.from_pretrained(model_folder_path)
pipeline.cuda()

# Load input image and run reconstruction
image = sample_utils.load_image(image_file_path)
outputs = pipeline.run(image)

# outputs contains:
#   - 'gaussian': 3D Gaussians
#   - 'radiance_field': radiance fields
#   - 'mesh': meshes
#   - 'pose': camera extrinsic and intrinsic parameters

# Save side-by-side comparison (input vs rendered)
render_rgb = render_utils.render_pose(outputs['gaussian'][0], outputs['pose'][0])['color'][0]
# Ensure image is in RGBA mode and matches the new image's size for alpha_composite
if image.mode != "RGBA":
    image = image.convert("RGBA")
input_rgb = Image.alpha_composite(Image.new("RGBA", image.size, (0, 0, 0, 255)), image)
input_rgb = np.array(input_rgb.resize((512, 512), Image.Resampling.LANCZOS).convert('RGB'))
imageio.imwrite(save_result_folder_path + 'sample.png', np.concatenate([input_rgb, render_rgb], axis=1))

# Save mesh and camera pose
save_mesh(
    all_outputs=outputs,
    poses=outputs.pop('pose'),
    output_dir=save_result_folder_path,
)
