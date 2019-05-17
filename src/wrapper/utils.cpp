#include <algorithm>
#include <fstream>

#include "utils.h"
#include "configs.h"

using namespace std;
using namespace Tn;
using namespace TinyYolo;

// Convert input image to a float vector, prepare for engine.
vector<float> 
prepareImage(cv::Mat &img)
{
	int c = INPUT_CHANNEL;
	int h = INPUT_HEIGHT;
	int w = INPUT_WIDTH;

	float scale = min(float(w) / img.cols, float(h) / img.rows);
	auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);

	cv::Mat rgb ;
	cv::cvtColor(img, rgb, CV_BGR2RGB);
	cv::Mat resized;
	cv::resize(rgb, resized, scaleSize, 0, 0, cv::INTER_CUBIC);
	cv::Mat cropped(h, w,CV_8UC3, 127);
	cv::Rect rect((w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height); 
	resized.copyTo(cropped(rect));

	cv::Mat img_float;
	if (c == 3)
		cropped.convertTo(img_float, CV_32FC3, 1/255.0);
	else
		cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

	//HWC TO CHW
	vector<cv::Mat> input_channels(c);
	cv::split(img_float, input_channels);

	vector<float> result(h * w * c);
	auto data = result.data();
	int channelLength = h * w;
	for (int i = 0; i < c; ++i) {
		memcpy(data, input_channels[i].data, channelLength * sizeof(float));
		data += channelLength;
	}

	return result;
}

// Extract information from the given detection result.
// If the detection result only has one.
TinyBbox 
postProcessImg1(Detection detections)
{
	int h = INPUT_HEIGHT;   //net h
	int w = INPUT_WIDTH;    //net w
	
	auto &b = detections.bbox;
	TinyBbox bbox = 
	{ 
		max(int((b[0] - b[2] / 2.f) * w), 0), //left
		min(int((b[0] + b[2] / 2.f) * w), w), //right
		max(int((b[1] - b[3] / 2.f) * h), 0), //top
		min(int((b[1] + b[3] / 2.f) * h), h), //bot
		detections.prob                           //score
	};

	return bbox;
}

// Extract information from many box.
vector<TinyBbox> 
postProcessImg2(cv::Mat &img, vector<Detection> &detections)
{
	int h = INPUT_HEIGHT;   //net h
	int w = INPUT_WIDTH;    //net w

	vector<TinyBbox> boxes;
	for(const auto &item : detections)
	{	      
		TinyBbox bbox = postProcessImg1(item);

		// Do threshold judgement, only box satify condition can it be pushed into result.
		if (bbox.left == 0 || bbox.right == w || bbox.top == 0 || bbox.bot == h)
			continue;
		if (bbox.left < 0.01* w || bbox.right > 0.99 * w || bbox.top < 0.01 * h || bbox.bot > 0.99 * h)
			continue;
		boxes.push_back(bbox);
	}

	return boxes;
}

// Auxiliary function helps write detection result to XML
void
WrBoxToXml (TinyBbox result, cv::String filename)
{
	fstream xmlfs;
	string imgName = split(split(string(filename.c_str()), '.').front(), '/').back();
	string xmlName = string(OUTPUTS_DIR) + imgName + ".xml";
	xmlfs.open (xmlName, fstream::out);
	xmlfs << "<annotation>" << endl
		  << "\t<filename>" << imgName << "</filename>" << endl
		  << "\t<size>" << endl
		  << "\t\t<width>" << INPUT_WIDTH << "</width>" << endl
		  << "\t\t<height>" << INPUT_HEIGHT << "</height>" << endl
		  << "\t</size>" << endl
		  << "\t<object>" << endl
		  << "\t\t<bndbox>" << endl
		  << "\t\t\t<xmin>" << result.left << "</xmin>" << endl
		  << "\t\t\t<ymin>" << result.top << "</ymin>" << endl
		  << "\t\t\t<xmax>" << result.right << "</xmax>" << endl
		  << "\t\t\t<ymax>" << result.bot << "</ymax>" << endl
		  << "\t\t</bndbox>" << endl
		  << "\t</object>" << endl
		  << "</annotation>";
	xmlfs.close();
}
