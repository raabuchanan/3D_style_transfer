/*
 * data_txt_reader.cpp
 *
 *  Created on: May 22, 2017
 *      Author: perry
 */


#include "vision_project/data_txt_reader.h"

#include <fstream>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <string>

unsigned int split(const std::string &txt, std::vector<std::string> &strs, char ch)
{
    size_t pos = txt.find( ch );
    unsigned int initialPos = 0;
    strs.clear();

    // Decompose statement
    while( pos != std::string::npos ) {
        strs.push_back( txt.substr( initialPos, pos - initialPos + 1 ) );
        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    strs.push_back( txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ) );

    return strs.size();
}


CameraData::CameraData(std::string camerastxt):camerastxt_(camerastxt){
	fillFromFile(camerastxt_);
}

void CameraData::fillFromFile(std::string camerastxt){
	camerastxt_ = camerastxt;

	std::ifstream input;
	input.open(camerastxt_.c_str());

	std::string line;
	while (std::getline(input,line)){

		if ( line[0] == '#'){
			continue;
		}

		std::vector<std::string> words;
	    split(line, words, ' ');

	    size_t index = 0;
	    size_t camera_id;
	    bool looking_for_ID = true;
	    while(looking_for_ID){
	    	try {
	    		camera_id = std::stoi(words[index]);
	    		looking_for_ID = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }

	    if (camera_id >= cameras_.size()){
	    	cameras_.resize(camera_id+1);
	    }

	    CameraModel& camera = cameras_[camera_id];

	    camera.id = camera_id;

	    bool looking_for_MODEL = true;
	    while(looking_for_MODEL){
	    	if (words[index]!=" "){
	    		looking_for_MODEL = false;
	    	}
	    	++index;
	    }

	    bool looking_for_WIDTH = true;
	    while(looking_for_WIDTH){
	    	try {
	    		camera.width = std::stoi(words[index]);
	    		looking_for_WIDTH = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }

	    bool looking_for_HEIGHT = true;
	    while(looking_for_HEIGHT){
	    	try {
	    		camera.height = std::stoi(words[index]);
	    		looking_for_HEIGHT = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }

	    bool looking_for_PARAM1 = true;
	    while(looking_for_PARAM1){
	    	try {
	    		camera.f = std::stod(words[index]);
	    		looking_for_PARAM1 = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }

	    bool looking_for_PARAM2 = true;
	    while(looking_for_PARAM2){
	    	if (words[index]!=" "){
	    		looking_for_PARAM2 = false;
	    	}
	    	++index;
	    }

	    bool looking_for_PARAM3 = true;
	    while(looking_for_PARAM3){
	    	if (words[index]!=" "){
	    		looking_for_PARAM3 = false;
	    	}
	    	++index;
	    }

	    bool looking_for_PARAM4 = true;
	    while(looking_for_PARAM4){
	    	try {
	    		camera.k = std::stod(words[index]);
	    		looking_for_PARAM4 = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }
	}

}

ImageData::ImageData(std::string imagestxt):imagestxt_(imagestxt){
	fillFromFile(imagestxt_);
}

void ImageData::fillFromFile(std::string imagestxt){
	imagestxt_ = imagestxt;

	std::ifstream input;
	input.open(imagestxt_.c_str());

	bool skip_line = false;
	std::string line;
	while (std::getline(input,line)){

		if ( line[0] == '#'){
			continue;
		}

		if ( skip_line ){
			skip_line = false;
			continue;
		}

		std::vector<std::string> words;
	    split(line, words, ' ');

	    size_t index = 0;
	    size_t image_id;
	    bool looking_for_ID = true;
	    while(looking_for_ID){
	    	try {
	    		image_id = std::stoi(words[index]);
	    		looking_for_ID = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }

	    if (image_id >= images_.size()){
	    	images_.resize(image_id+1);
	    }

	    ImageModel& image = images_[image_id];

	    image.id = image_id;

	    skip_line = true;


	    bool looking_for_QW = true;
	    while(looking_for_QW){
	    	try {
	    		image.quat.w() = std::stod(words[index]);
	    		looking_for_QW = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }


	    bool looking_for_QX = true;
	    while(looking_for_QX){
	    	try {
	    		image.quat.x() = std::stod(words[index]);
	    		looking_for_QX = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }

	    bool looking_for_QY = true;
	    while(looking_for_QY){
	    	try {
	    		image.quat.y() = std::stod(words[index]);
	    		looking_for_QY = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }

	    bool looking_for_QZ = true;
	    while(looking_for_QZ){
	    	try {
	    		image.quat.z() = std::stod(words[index]);
	    		looking_for_QZ = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }


	    bool looking_for_TX = true;
	    while(looking_for_TX){
	    	try {
	    		image.trans[0] = std::stod(words[index]);
	    		looking_for_TX = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }


	    bool looking_for_TY = true;
	    while(looking_for_TY){
	    	try {
	    		image.trans[1] = std::stod(words[index]);
	    		looking_for_TY = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }

	    bool looking_for_TZ = true;
	    while(looking_for_TZ){
	    	try {
	    		image.trans[2] = std::stod(words[index]);
	    		looking_for_TZ = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }

	    bool looking_for_CAMERA_ID = true;
	    while(looking_for_CAMERA_ID){
	    	try {
	    		image.camera_id = std::stoi(words[index]);
	    		looking_for_CAMERA_ID = false;
		    	++index;
			} catch (const std::exception& e){
		    	++index;
			}
	    }


	    bool looking_for_FILE = true;
	    while(looking_for_FILE){
	    	if (words[index] != " "){
	    		looking_for_FILE = false;
	    		image.image_name = words[index];
	    	}
	    	++index;
	    }
	}

}

