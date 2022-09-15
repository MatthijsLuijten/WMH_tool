export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export THEANO_FLAGS='floatX=float32, device=gpu0, optimizer_including=cudnn, mode=FAST_RUN, optimizer=fast_compile'

python main.py /home/mohsen/WMH/WMH_tool/train_and_test/description.lpd
