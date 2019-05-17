#ifndef _TINY_YOLO_CONFIGS_H_
#define _TINY_YOLO_CONFIGS_H_


namespace TinyYolo
{
	static constexpr int CHECK_COUNT = 3;
	static constexpr float IGNORE_THRESH = 0.00001f;

	struct TinyYoloKernel
	{
		int width;
		int height;
		float anchors[CHECK_COUNT*2];
	};

	TinyYoloKernel yolo1 = {
		20,
		12,
		{0.1947, 0.1971, 0.3245, 0.4063, 0.8269, 0.7668}
	};

	TinyYoloKernel yolo2 = {
		40,
		24,
		{0.0241, 0.0336, 0.0553, 0.0649, 0.0890, 0.1394}
	};
}

#endif
