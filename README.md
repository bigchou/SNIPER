# R-FCN-3000

# Installation Guide

git clone -b cvpr3k --single-branch https://github.com/bigchou/SNIPER.git

see https://docs.google.com/document/d/1MutrvFh9y3RMx4yuUDfN2pjTOvVt7deckndP-Ym106I/edit?usp=sharing

~~~~

# The MxNet-SNIPER installation depends on CUDA, CuDNN, OpenCV, and OpenBLAS
# We assume you've install CUDA and CUDNN.

# Step 1. Install OpenBLAS
sudo apt-get install gcc-5 g++-5
sudo apt-get install libopenblas-dev

# Step 2. Install OpenCV
mkdir opencv && cd opencv
wget https://cmake.org/files/v3.6/cmake-3.6.1-Linux-x86_64.tar.gz
tar -zxvf cmake-3.6.1-Linux-x86_64.tar.gz
alias cmake="/media/iis/ssdx16/cmake-3.6.1-Linux-x86_64/bin/cmake"
wget https://codeload.github.com/opencv/opencv/zip/2.4.13
unzip 2.4.13
cd opencv-2.4.13
mkdir release
cd release/
cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j10
sudo make install
vim ~/.bashrc
# Then, add "export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/" into ~/.bashrc
source ~/.bashrc
pkg-config --cflags --libs opencv # check opencv installation
cd ../../../

# Step 3. Install MxNet-SNIPER
git clone --recursive https://github.com/mahyarnajibi/SNIPER.git
cd SNIPER/SNIPER-mxnet
make -j 8 USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/[CUDA_PATH] # E.g. [CUDA_PATH] is "/usr/local/cuda-10.1" for CUDA 10.1
cd ..
python -m pip install --upgrade pip
python -m pip install Cython
python -m pip install numpy==1.15.1
python -m pip install opencv-python==4.2.0.32
bash scripts/compile.sh
python -m pip install -r requirements.txt

# Step 4. Install Python binding
cd SNIPER-mxnet
python -m pip install --user -e ./python
python -c 'import mxnet; print(mxnet.__version__)' # check the mxnet python binding. E.g. output is 1.2.0
pip install futures
cd ../scripts
bash download_sniper_autofocus_detectors.sh
cd ..
python demo.py # check whether the mxnet-sniper works without errors

# Step 5. Install R-FCN-3000
cd ..
mkdir RFCN3000 && cd RFCN3000
git clone -b cvpr3k --single-branch https://github.com/bigchou/SNIPER.git
cd SNIPER
bash scripts/compile.sh
python run.py --mode draw --topk 5 --inputdir ooo # check R-FCN-3000 installation
more bbox_info.json # see the location predictions
~~~~

#### Note. this repo is for python2 only


# Execution

### make dataset

~~~~
python mydemo.py
~~~~

### predict real-shot images

~~~~
# Drawing
python run.py --mode draw --topk 5
# Cropping
python run.py --mode crop --topk 1
# Disable non-maximum suppression (NMS)
python run.py --nonms --mode crop --topk 1
~~~~

#### Note.

1. [OPTION] Please donwload these images https://drive.google.com/drive/folders/1KU6bBvjYIDDJh-ry8t5BOKlthchFD7G1?usp=sharing and put it in <b>demo/</b>


# Visualize detection results using DetVisGUI provided by <a href="https://github.com/Chien-Hung/DetVisGUI">Chien-Hung</a>

<img src="demo/snapshot.jpg" height= 300px>


# Results
see <b>myoutput</b> for more details.

<table>
 <tr>
 <td><img src="myoutput/2.jpg" height = 100px></td>
 <td><img src="myoutput/7.jpg" height = 100px></td>
 <td><img src="myoutput/26.jpg" height = 100px></td>
 <td><img src="myoutput/25.jpg" height = 100px></td>
 <td><img src="myoutput/27.jpg" height = 100px></td>
 <td><img src="myoutput/18.jpg" height = 100px></td>
 <td><img src="myoutput/17.jpg" height = 100px></td>
 <td><img src="myoutput/3.jpg" height = 100px></td>
 <td><img src="myoutput/14.jpg" height = 100px></td>
 <td><img src="myoutput/5.jpg" height = 100px></td>
 <td><img src="myoutput/8.jpg" height = 100px></td>
 </tr>
</table>




# Note.

[1] Remember to do <b>git push origin cvpr3k</b>

[2] Extract files using <b>cat yahoo.part* | tar zxvf -</b>

[3] zip -r -qq pchomefcncrop.zip products-pchome products-pchome.json
