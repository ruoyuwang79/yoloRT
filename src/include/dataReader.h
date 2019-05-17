#ifndef _DATA_READER_H_
#define _DATA_READER_H_

#include <vector>
#include <list>
#include <string>

namespace Tn
{
	struct Source
	{
		std::string fileName;
		int label;
	};

	struct Bbox
	{
		int classId;
		int left;
		int right;
		int top;
		int bot;
		float score;
	};

	struct TinyBbox
	{
		int left;
		int right;
		int top;
		int bot;
		float score;
	};

	std::list<std::string> readFileList(const std::string &fileName);
	std::list<Source> readLabelFileList(const std::string &fileName);    
	std::vector<std::string> split(const std::string &str, char delim);
	std::tuple<std::list<std::string>, std::list<std::vector<Bbox>>> readObjectLabelFileList(const std::string &fileName);
}

#endif
