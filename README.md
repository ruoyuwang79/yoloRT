# YoloRT
  - An inference network
  - Edge inference with low power
  - TensorRT speedup

## Requirement
	- openCV
	- CUDA
	- TensorRT

## File Tree
	|--- yoloRT	
			|--- src (all source files)
				  |- aux (auxiliary functions)
				  |- include (all headers)
				  |- wrapper (tensorRT engine generator)
			|--- CMakeList.txt
			|--- install.sh (use this script to build and install)
			|--- run.sh (use this script to run)
	|--- images (should be created by yourself, contains all .jpg images you want to inference)
			|--- result (evaluation results)
					|- time
						|- alltime.txt (output fps)
					|- xml
						|- DeepZS (our team name directory, all detection will write as .xml to here)
			|--- ...*.jpg

## Configuration

In `src/include/config.h`, there are several configuration options.  
	1. `ENGINE_DIR` declares the TensorRT engine directory, no matter save or load.  
	2. `INPUT_PROTOTXT` declares the `.prototxt` position, which define the network.  
	3. `INPUT_CAFFEMODEL` declares the `.caffemodel` position, also define the network.  
	4. `INPUT_IMAGES` declares the input images directory, change it to directory as you wish.  
	5. `EVAL_NAME` declares the 'fps' values output file.  
	6. `OUTPUTS_DIR` declares the '.xml' result output directory.  
	7. `MODE` declares engine mode, can be 'FLOAT16', 'FLOAT32' or 'INT8'.  
	8. `MODE_NAME` used to generate engine name and reuse them.  
	9. `OUTPUTS` the last layer (output layer) name of the given model. If you want to change it, change the `.prototxt` as well.  
	10. `INPUT_WIDTH` declares the input images width.  
	11. `INPUT_HEIGHT` declares the input images height.  
	12. `LOAD_FROM_ENGINE` defines the program's work mode, '1' indicates load from `ENGINE_DIR`, others indicates generate a new engine by using given `.prototx` and `.caffemodel`, and save new engine to `.ENGINE_DIR`.  

## Data Preparation 

```
mkdir models
```

Put your `.prototxt` file and `.caffemodel` file in the models directory.
Also, if you want to reuse the generated engine, it will also be stored in this directory.

## Build and Install

```
./install.sh
```  

You will find a executable file called 'yoloRT' under the build directory.

## Run Inference

```
./run.sh
```

It will take a long time for building a new engine.
If load the given engine, it will be more faster, the program will only print the final fps in the terminal.
