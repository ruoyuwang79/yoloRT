#include "TinyYoloConfigs.h"
#include "TinyYoloLayer.h"

using namespace TinyYolo;


namespace nvinfer1
{
	TinyYoloLayerPlugin::TinyYoloLayerPlugin(const int cudaThread /*= 512*/):mThreadCount(cudaThread)
	{
		mYoloKernel.clear();
		mYoloKernel.push_back(yolo1);
		mYoloKernel.push_back(yolo2);
		mKernelCount = mYoloKernel.size();
	}

	TinyYoloLayerPlugin::~TinyYoloLayerPlugin()
	{
		if(mInputBuffer)
			CUDA_CHECK(cudaFreeHost(mInputBuffer));

		if(mOutputBuffer)
			CUDA_CHECK(cudaFreeHost(mOutputBuffer));
	}
	// create the plugin at runtime from a byte stream
	TinyYoloLayerPlugin::TinyYoloLayerPlugin(const void* data, size_t length)
	{
		using namespace Tn;
		const char *d = reinterpret_cast<const char *>(data);
		read(d, mThreadCount);
		read(d, mKernelCount);
		mYoloKernel.resize(mKernelCount);
		auto kernelSize = mKernelCount*sizeof(TinyYoloKernel);
		memcpy(mYoloKernel.data(),d,kernelSize);
		d += kernelSize;
	}
	void TinyYoloLayerPlugin::serialize(void* buffer)
	{
		using namespace Tn;
		char* d = static_cast<char*>(buffer);
		write(d, mThreadCount);
		write(d, mKernelCount);
		auto kernelSize = mKernelCount*sizeof(TinyYoloKernel);
		memcpy(d,mYoloKernel.data(),kernelSize);
		d += kernelSize;
	}
	size_t TinyYoloLayerPlugin::getSerializationSize()
	{  
		return sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(TinyYolo::TinyYoloKernel) * mYoloKernel.size();
	}
	int TinyYoloLayerPlugin::initialize()
	{ 
		int totalCount = 0;
		for(const auto& yolo : mYoloKernel)
			totalCount += (LOCATIONS + 1) * yolo.height * yolo.width * CHECK_COUNT;
		CUDA_CHECK(cudaHostAlloc(&mInputBuffer, totalCount * sizeof(float), cudaHostAllocDefault));
		CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, sizeof(float) + totalCount * sizeof(Detection), cudaHostAllocDefault));
		//CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, sizeof(Detection), cudaHostAllocDefault));
		return 0;
	}

	Dims TinyYoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
	{
		//output the result to channel
				
		int totalCount = 0;
		for(const auto& yolo : mYoloKernel)
			totalCount += yolo.width*yolo.height * CHECK_COUNT * sizeof(Detection) / sizeof(float);
		
		//int totalCount = sizeof(Detection) / sizeof(float);
		return Dims3(totalCount, 1, 1);
	}
	/*
	void TinyYoloLayerPlugin::forwardCpu(const float*const * inputs, float* outputs, cudaStream_t stream)
	{
		auto Logist = [=](float data){
			return 1./(1. + exp(-data));
		};

		CUDA_CHECK(cudaStreamSynchronize(stream));
		int i = 0;
		float* inputData = (float *)mInputBuffer; 
		for(const auto& yolo : mYoloKernel)
		{
			int size = (LOCATIONS + 1) * yolo.width*yolo.height * CHECK_COUNT;
			CUDA_CHECK(cudaMemcpyAsync(inputData, inputs[i], size * sizeof(float), cudaMemcpyDeviceToHost, stream));
			inputData += size;
			++ i;
		}
		inputData = (float *)mInputBuffer;
		std::vector <Detection> result;
		float maxProb = 0;
		for (const auto& yolo : mYoloKernel)
		{
			int stride = yolo.width*yolo.height;
			for (int y = 0; y < yolo.height; ++y) {
				for (int x = 0 ; x < yolo.width; ++x) {
					for (int n = 0; n < CHECK_COUNT; ++n){
						int index = x + yolo.width * (y + yolo.height * (LOCATIONS + 1) * n);
						float objProb = Logist(inputData[index]);
						if (objProb > maxProb) {
							Detection det;
							maxProb = objProb;
							det.prob = maxProb;
							det.bbox[0] = (x + Logist(inputData[index + 1 * stride])) / yolo.width;
							det.bbox[1] = (y + Logist(inputData[index + 2 * stride])) / yolo.height;
							det.bbox[2] = exp(inputData[index + 3 * stride]) * yolo.anchors[2 * n];
							det.bbox[3] = exp(inputData[index + 4 * stride]) * yolo.anchors[2 * n + 1];
							if (!result.empty())
								result.pop_back();
							result.emplace_back(det);           
						}
					}
				}
			}
			inputData += (LOCATIONS + 1) * stride * CHECK_COUNT;
		}
		auto data = (float *)mOutputBuffer;
		memcpy(data, result.data(), sizeof(Detection));
		CUDA_CHECK(cudaMemcpyAsync(outputs, mOutputBuffer, sizeof(Detection), cudaMemcpyHostToDevice, stream));
	}
	*/

	__device__ float LogistTiny(float data){ return 1./(1. + exp(-data)); };

	__global__ void CalDetectionTiny(const float *input, float *output,int noElements, 
			int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2]) {
 
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= noElements) return;

		int stride = yoloWidth*yoloHeight;
		for (int k = 0;k < CHECK_COUNT; ++k )
		{
			int beginIdx = (LOCATIONS + 1)* stride *k + idx;
			int objIndex = beginIdx;
			
			//check objectness
			float objProb = LogistTiny(input[objIndex]);   
			if(objProb <= IGNORE_THRESH)
				continue;

			int row = idx / yoloWidth;
			int cols = idx % yoloWidth;
			
			//classes
			int resCount = (int)atomicAdd(output,1);
			char* data = (char * )output + sizeof(float) + resCount*sizeof(Detection);
			//char* data = (char * )output + sizeof(Detection);
			Detection* det =  (Detection*)(data);

			//Location
			det->bbox[0] = (cols + LogistTiny(input[beginIdx+stride]))/ yoloWidth;
			det->bbox[1] = (row + LogistTiny(input[beginIdx+2*stride]))/ yoloHeight;
			det->bbox[2] = exp(input[beginIdx+3*stride]) * anchors[2*k];
			det->bbox[3] = exp(input[beginIdx+4*stride]) * anchors[2*k + 1];
			det->prob = objProb;
		}
	}
   
	void TinyYoloLayerPlugin::forwardGpu(const float *const * inputs,float * output,cudaStream_t stream) {
		int numElem;
		void* devAnchor;
		size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
		CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));

		//first detect count init 0
		CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
		for (unsigned int i = 0;i< mYoloKernel.size();++i)
		{
			const auto& yolo = mYoloKernel[i];
			numElem = yolo.width*yolo.height;

			//copy anchor to device
			CUDA_CHECK(cudaMemcpy(devAnchor,yolo.anchors,AnchorLen,cudaMemcpyHostToDevice));

			CalDetectionTiny<<< (yolo.width*yolo.height + mThreadCount - 1) / mThreadCount, mThreadCount>>>
					(inputs[i],output, numElem, yolo.width, yolo.height, (float *)devAnchor);
		}
		CUDA_CHECK(cudaFree(devAnchor));
	}


	int TinyYoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
	{
		assert(batchSize == 1);
		
		//GPU
		forwardGpu((const float *const *)inputs,(float *)outputs[0],stream);

		//CPU
		//forwardCpu((const float *const *)inputs,(float *)outputs[0],stream);
		return 0;
	};

}

