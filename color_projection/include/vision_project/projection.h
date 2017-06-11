/*
 * projection.h
 *
 *  Created on: May 21, 2017
 *      Author: perry
 */

#ifndef INCLUDE_VISION_PROJECT_PROJECTION_H_
#define INCLUDE_VISION_PROJECT_PROJECTION_H_

#include <pcl/common/transforms.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;

// This class is used to store a matrix, where each element is of
// type "val". In our code, we use it to store the 3d points that
// are "caught" in a pixel of an image.
template<typename val>
class MatrixOfVectors {
public:
	MatrixOfVectors(size_t height, size_t width) :
			height_(height), width_(width) {

		array = new std::vector<val>*[height_ * width_];

		for (size_t i = 0; i < width_; ++i) {
			for (size_t j = 0; j < height_; j++) {
				array[j * width + i] = new std::vector<val>;
			}
		}
	}
	~MatrixOfVectors() {

		for (size_t i = 0; i < width_; ++i) {
			for (size_t j = 0; j < height_; j++) {
				delete array[j * width_ + i];
			}
		}
		delete array;

	}

	std::vector<val>& at(size_t i, size_t j) {
		return *array[j * width_ + i];
	}

	const std::vector<val>& get(size_t i, size_t j) const {
		return *array[j * width_ + i];
	}

	const size_t getHeight() const {
		return height_;
	}

	const size_t getWidth() const {
		return width_;
	}

private:

	const size_t height_;
	const size_t width_;
	std::vector<val>** array;

};
//
//template<typename PointT >
//MatrixOfVectors< PointT* > projectPointsPlus(int height, int width, double f, double k1,
//		boost::shared_ptr<pcl::PointCloud<PointT> > pointcloud);
//
//template<typename PointT >
//void displayMatOfVecGray(const MatrixOfVectors< PointT* >& in, const std::string& window);
//
//template<typename PointT >
//void displayMatOfVecColor(const MatrixOfVectors< PointT* >& in, const std::string& window);

//// TEMPLATE FUNCTION DEFINITIONS

template<typename PointT>
MatrixOfVectors<PointT*> projectPointsPlus(int height, int width, double f,
		double k1, boost::shared_ptr<pcl::PointCloud<PointT> > pointcloud) {

	MatrixOfVectors<PointT*> output(height, width);

	unsigned int cx = width / 2;
	unsigned int cy = height / 2;

	for (size_t i = 0; i < pointcloud->size(); ++i) {
		double x_prime = pointcloud->at(i).x / pointcloud->at(i).z;
		double y_prime = pointcloud->at(i).y / pointcloud->at(i).z;

		double r2 = (x_prime * x_prime) + (y_prime * y_prime);

		double x_primeprime = x_prime * (1 + k1 * r2);
		double y_primeprime = y_prime * (1 + k1 * r2);

		double u = f * x_primeprime;
		double v = f * y_primeprime;

		size_t rounded_u = u + cx;
		size_t rounded_v = v + cy;

		if ((rounded_u >= 0 && rounded_u < output.getWidth())
				&& (rounded_v >= 0 && rounded_v < output.getHeight())) {
			output.at(rounded_u, rounded_v).push_back(&(pointcloud->at(i)));
		}
	}

	return output;
}

template<typename PointT>
void displayMatOfVecGray(const MatrixOfVectors<PointT*>& in,
		const std::string& window) {

	Mat image(in.getHeight(), in.getWidth(), CV_8U);
	image.setTo(0);

	for (size_t y = 0; y < in.getHeight(); ++y) {
		for (size_t x = 0; x < in.getWidth(); ++x) {
			if (!in.get(x, y).empty()) {
				image.at<unsigned char>(y, x) = std::min<size_t>(
						in.get(x, y).size() * 5, 255);
			}
		}
	}

	imshow(window, image);
}

template<typename PointT>
void displayMatOfVecColor(const MatrixOfVectors<PointT*>& in,
		const std::string& window) {

	Mat image(in.getHeight(), in.getWidth(), CV_8UC3);
	image.setTo(Vec3b(0, 0, 0));

	for (size_t y = 0; y < in.getHeight(); ++y) {
		for (size_t x = 0; x < in.getWidth(); ++x) {
			PointT* closest_point = NULL;
			for (PointT* cur_point : in.get(x, y)) {
				if (!closest_point) {
					closest_point = cur_point;
				} else {
					if (closest_point->z > cur_point->z) {
						closest_point = cur_point;
					}
				}
			}

			if (closest_point) {
				Vec3b& pixel = image.at<Vec3b>(y, x);

				uint32_t rgb = *reinterpret_cast<int*>(&closest_point->rgb);

				pixel[2] = (rgb >> 16) & 0x0000ff;
				pixel[1] = (rgb >> 8) & 0x0000ff;
				pixel[0] = (rgb) & 0x0000ff;
			}
		}
	}

	imshow(window, image);
}

template<typename PointT>
void recolor(Mat image, MatrixOfVectors<PointT*>& in) {

	if (image.cols != in.getWidth() || image.rows != in.getHeight()) {
		std::cerr << "Dimensions do not match" << std::endl;
		std::cerr << "Image width = "<< image.cols <<"   matofvec width = "<< in.getWidth() << std::endl;
		std::cerr << "Image height = "<< image.rows <<"   matofvec height = "<< in.getHeight() << std::endl;

		return;
	}

	std::vector< std::vector < double > > closest_distance;


	for (int y = 0; y < in.getHeight(); ++y) {

		closest_distance.push_back( std::vector<double>() );

		for (int x = 0; x < in.getWidth(); ++x) {
			closest_distance[y].push_back(std::numeric_limits<double>::max());

			int margin = 1;

			for (int v = -margin; v <=margin; ++v){
				for (int u = -margin; u <= margin; ++u){

					int true_y = y+v;
					int true_x = x+u;

					if (true_y < 0 || true_y >= in.getHeight()){
						continue;
					}
					if (true_x < 0 || true_x >= in.getWidth()){
						continue;
					}

					PointT* closest_point = NULL;
					for (PointT* cur_point : in.get(true_x, true_y)) {
						if (!closest_point) {
							closest_point = cur_point;
						} else {
							if (closest_point->z > cur_point->z) {
								closest_point = cur_point;
							}
						}
					}

					if (closest_point){
						closest_distance[y][x] = std::min<double>(closest_distance[y][x], closest_point->z);
					}

				}
			}


		}
	}


	for (size_t y = 0; y < in.getHeight(); ++y) {
		for (size_t x = 0; x < in.getWidth(); ++x) {

			if (in.get(x, y).empty()){
				continue;
			}

//			PointT* closest_point = NULL;
//			for (PointT* cur_point : in.get(x, y)) {
//				if (!closest_point) {
//					closest_point = cur_point;
//				} else {
//					if (closest_point->z > cur_point->z) {
//						closest_point = cur_point;
//					}
//				}
//			}

			double threshold = 0.01;
			Vec3b pixel = image.at<Vec3b>(y,x);
			for (PointT* cur_point : in.get(x, y)) {
//				if (cur_point->z - closest_point->z < threshold ) {
				if (cur_point->z - closest_distance[y][x] < threshold ) {

					int red = ((int) (cur_point->a) * ((int) (cur_point->r)) + (int) pixel[2])/((int) (cur_point->a) + 1);
					int green = ((int) (cur_point->a) * ((int) (cur_point->g)) + (int) pixel[1])/((int) (cur_point->a) + 1);
					int blue = ((int) (cur_point->a) * ((int) (cur_point->b)) + (int) pixel[0])/((int) (cur_point->a) + 1);


					cur_point->r = red;
					cur_point->g = green;
					cur_point->b = blue;

					++cur_point->a;

				}
			}

		}
	}
}

#endif /* INCLUDE_VISION_PROJECT_PROJECTION_H_ */
