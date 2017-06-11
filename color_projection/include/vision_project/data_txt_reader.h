/*
 * data_txt_reader.h
 *
 *  Created on: May 22, 2017
 *      Author: perry
 */

#ifndef INCLUDE_VISION_PROJECT_DATA_TXT_READER_H_
#define INCLUDE_VISION_PROJECT_DATA_TXT_READER_H_

#include <Eigen/Dense>
#include <vector>

struct CameraModel{

	double f;
	double k;

	unsigned int id;

	size_t width;
	size_t height;

};

struct ImageModel{

	size_t camera_id;
	int id;

	Eigen::Quaterniond quat;
	Eigen::Vector3d trans;

	std::string image_name;

};

class CameraData{

public:

	CameraData(std::string camerastxt);
	~CameraData(){}

	void fillFromFile(std::string camerastxt);

	std::vector<CameraModel> cameras_;

private:
	std::string camerastxt_;

	size_t max_size;

};

class ImageData{
public:

	ImageData(std::string imagestxt);
	~ImageData(){};

	void fillFromFile(std::string imagestxt);

	std::vector<ImageModel> images_;

private:
	std::string imagestxt_;


};



#endif /* INCLUDE_VISION_PROJECT_DATA_TXT_READER_H_ */
