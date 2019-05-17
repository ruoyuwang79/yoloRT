#ifndef __UTILS_H_
#define __UTILS_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <list>

#include "dataReader.h"
#include "TinyYoloLayer.h"

std::vector<float> prepareImage (cv::Mat &img);
Tn::TinyBbox postProcessImg1 (TinyYolo::Detection detections);
std::vector<Tn::TinyBbox> postProcessImg2 (cv::Mat &img, std::vector<TinyYolo::Detection> &detections);
void WrBoxToXml (Tn::TinyBbox result, cv::String filename);

#endif
