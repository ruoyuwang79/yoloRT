# YoloRT
	- An inference network
	- Edge inference with low power
	- TensorRT speedup

## Requirement
	- openCV
	- CUDA
	- TensorRT

## Build and Install

```
mkdir build
cd build && cmake .. && make install
```  

You will find a executable file called 'yoloRT' under the build directory.

## Data Preparation 

```
mkdir models
```

Put your `.prototxt` file and `.caffemodel` file in the models directory.
Also, if you want to reuse the generated engine, it will also be stored in this directory.

## Configuration

Only need to change the configuration in the `src/include/config.h`

## Run Inference

In the `build` directory, run
```
./yoloRT
```
