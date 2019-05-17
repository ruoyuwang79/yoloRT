#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>

#include "utils.h"
#include "configs.h"

#include "TrtNet.h"
#include "TinyYoloLayer.h"
#include "dataReader.h"

using namespace std;
using namespace Tn;
using namespace TinyYolo;

int main(int argc, char* argv[])
{
#if LOAD_FROM_ENGINE == 1
	// if we should load an exist engine, get its name.
	string saveName = string(ENGINE_DIR) + "yolov3_tiny_" + string(MODE_NAME) + ".engine";
	trtNet net(saveName);
#else
	// else, we should generate a new engine from the given prototxt and model.
	string deployFile = INPUT_PROTOTXT;
	string caffemodelFile = INPUT_CAFFEMODEL;
	vector<vector<float>> calibData;
	string outputNodes = OUTPUTS;
	auto outputNames = split(outputNodes, ',');
	trtNet net(deployFile, caffemodelFile, outputNames, calibData, MODE);
	cout << "save Engine..." << saveName << endl;
	net.saveEngine(saveName);
#endif
	// The output buffer.
	int outputCount = net.getOutputSize() / sizeof(float);
	unique_ptr<float[]> outputData(new float[outputCount]);

	// The vector contains all input files name.
	vector<cv::String> fileNames;
	cv::glob(cv::String(INPUT_IMAGES) + cv::String("*.jpg"), fileNames, false);
	int fileNumbers = fileNames.size();
	
	// Used to record inference time.
	float time;
	float totaltime = 0;
	float totalfps = 0;

	for (const auto &filename : fileNames)
	{
		// read in every file in the name list.
		cv::Mat img = cv::imread(filename);
		vector<float> inputData = prepareImage(img);
		if (!inputData.data())
			continue;
		
		// the method 'doInference' will return the time it consumed for the given image in ms.
		time = net.doInference(inputData.data(), outputData.get());
		totaltime += time;

		//Get Output    
		auto output = outputData.get();

		//first detect count
		int count = output[0];
		
		//later detect result
		vector<Detection> result;
		result.resize(count);
		memcpy(result.data(), &output[1], count * sizeof(Detection));		

		auto boxes = postProcessImg2(img, result);

		// No detection result
		if (boxes.size() == 0)
			continue;

		// Get the largest probability box
		float maxProb = 0;
		TinyBbox Max;
		for (const auto &data : boxes) {
			if (data.score > maxProb) {
				Max.score = data.score;
				Max.left = data.left;
				Max.right = data.right;
				Max.top = data.top;
				Max.bot= data.bot;
				maxProb = Max.score;
			}
		}

		// Write to xml
		WrBoxToXml(Max, filename);
	}

	// Because time is in ms, should multiply 1000
	totalfps = fileNumbers / totaltime * 1000;
	cout << "total fps: " << totalfps << endl;

	// write fps to the target file
	fstream fs;
	fs.open(EVAL_NAME, fstream::out);
	fs << "total fps: " << totalfps << endl;
	fs.close();

	return 0;
}
