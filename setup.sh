cd ..
git clone --depth 1 --recursive -b v2.8.3 https://github.com/Dao-AILab/flash-attention.git
git clone --depth 1 --recursive https://github.com/NVlabs/nvdiffrast.git
git clone --depth 1 --recursive https://github.com/hbb1/diffoctreerast.git
git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git

conda install -y conda-forge::libjpeg-turbo

pip install ninja

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install imageio imageio-ffmpeg tqdm easydict \
  opencv-python-headless scipy ninja rembg onnxruntime \
  trimesh open3d xatlas pyvista pymeshfix igraph transformers \
  tensorboard pandas lpips iopath spconv-cu120

pip install gradio==4.44.1 gradio_litmodel3d==0.0.1

CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 --no-cache-dir --no-build-isolation

pip install xformers==0.0.29.post3 \
  --index-url https://download.pytorch.org/whl/cu124

pip install kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html

pip install "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
pip install bpy==4.3.0 --extra-index-url https://download.blender.org/pypi/
pip install "git+https://github.com/hbb1/my_blendertoolbox.git"

cd flash-attention
python setup.py install

cd ../nvdiffrast
python setup.py install

cd ../diffoctreerast
python setup.py install

cd ../mip-splatting
pip install submodules/diff-gaussian-rasterization/ --no-cache-dir --no-build-isolation

cd ../pytorch3d
python setup.py install

cd ../cupid
pip install extensions/vox2seq --no-cache-dir --no-build-isolation
