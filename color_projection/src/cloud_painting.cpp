/*
 * cloud_painting.cpp
 *
 *  Created on: May 23, 2017
 *      Author: Perry Franklin
 *
 *  This file creates an executable that can be used to color a point
 *  cloud using images. See the comment above "main" for more details.
 */

#include <pcl/io/ply_io.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

#include "vision_project/projection.h"

#include <algorithm>
#include <thread>

#include "vision_project/data_txt_reader.h"
#include "vision_project/pcl_simple_viewer.h"

typedef pcl::PointXYZRGB pointtype;

const double PI = 3.14159265359;

using namespace cv;

// The MyViewer class is just a convenience class to provide a parallel
// pcl viewer, ie one that will continue updates even while other code is
// running.
class MyViewer {
public:

	MyViewer(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_) :
			viewer(viewer_) {

		view_thread = new std::thread(&MyViewer::viewthread, this);

	}
	~MyViewer() {
		view_thread->join();
		delete view_thread;
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

private:

	std::thread* view_thread;

	void viewthread() {
		while (!viewer->wasStopped()) {
			viewer->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	}
};

// This function colors a point cloud using a provide image and the camera data
// from ColMap.
template<typename PointT>
void colorPointCloud(ImageModel im, CameraModel cm, Mat color_image,
		boost::shared_ptr<pcl::PointCloud<PointT>>& cloud_in_out) {

	Eigen::Quaterniond quat = im.quat;
	Eigen::Translation3d trans;
	trans.x() = im.trans[0];
	trans.y() = im.trans[1];
	trans.z() = im.trans[2];

	Eigen::Transform<double, 3, Eigen::Affine> camera_to_pointcloud = trans
			* quat;

	boost::shared_ptr<pcl::PointCloud<PointT> > transformed_cloud(
			new pcl::PointCloud<PointT>());
	pcl::transformPointCloud(*cloud_in_out, *transformed_cloud,
			camera_to_pointcloud.matrix());

	MatrixOfVectors<PointT*> projection = projectPointsPlus<PointT>(cm.height,
			cm.width, cm.f, cm.k, transformed_cloud);

	recolor(color_image, projection);

	pcl::transformPointCloud(*transformed_cloud, *cloud_in_out,
			camera_to_pointcloud.inverse());
}

// This program is designed to be used with a point cloud generated from Colmap, with color
// images corresponding to those used to create the point cloud. The cameras and images.txt
// files are used to set the locations and camera parameters for each image. The coloring
// folder contains all the images with the color to be applied, and they must be named
// the same as the images used to generate the point cloud.

// Usage of the program is as follows:
// The PLY file is loaded
// The program will ask if you wish to use all the images to color the point cloud, or just a
// few hand-selected ones.
// If you choose to use all of them, it iterates through all of them and colors the point cloud
// incrementally.
// If you choose a few, you will be prompted to input a index number. Note that this index number
// is of the index that the images are loaded, NOT necessarily a number corresponding to a file
// name.
// Once the program is completed (or you've quit), it asks for a file name to save the point cloud
// If you input anything that does not have a .ply at the end, it quits. If you use a file name
// that ends in .ply, it will save to that file.
//
// NOTE: due to the way this is programmed, if you use more than 255 images, unpredictable behavior
// may occur. Sorry.
int main(int argc, char** argv) {
	pcl::PointCloud<pointtype>::Ptr cloud(new pcl::PointCloud<pointtype>);

	if (argc <= 4) {
		std::cerr
				<< " Usage: ./cloud_painting PLY_FILE CAMERAS.TXT IMAGES.TXT COLORINGFOLDER"
				<< std::endl;

		std::cerr << " But you only provided " << argc << " arguments"
				<< std::endl;
		return (-1);
	}

	if (pcl::io::loadPLYFile(argv[1], *cloud) == -1) //* load the file
			{
		PCL_ERROR("Couldn't read file test_pcd.pcd \n");
		return (-1);
	}

	for (auto& element : cloud->points) {
		element.r = 0;
		element.g = 0;
		element.b = 0;
		element.a = 0;
	}

	CameraData camera_data(argv[2]);
	ImageData image_data(argv[3]);

	std::string color_folder = argv[4];

	namedWindow("StyleImage", WINDOW_AUTOSIZE);

// ----------------------  PCL VISUALIZATION ----------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer =
			vision_project::createSimpleViewer(cloud, "3D viewer");

	MyViewer myview(viewer);

	std::vector<pcl::visualization::Camera> cameras;
	viewer->getCameras(cameras);

	pcl::visualization::Camera camera = cameras[0];

	std::cout << "Input 'all' to just process all the images available"
			<< std::endl;
	std::string option;
	std::cin >> option;

	if (option != "all") {

		std::cout << "By the way, I have information for "
				<< image_data.images_.size() << " images" << std::endl
				<< std::endl;

		while (true) {

			std::cout << "Please input the next image index, or q to quit"
					<< std::endl;

			std::string next_image_id_str;
			cin >> next_image_id_str;

			if (next_image_id_str == "q") {
				break;
			}

			int next_image_id;
			try {
				next_image_id = std::stoi(next_image_id_str);
			} catch (const std::invalid_argument& e) {
				std::cout << "That's not a number I can read...." << std::endl
						<< std::endl;
			}

			if (!(next_image_id < image_data.images_.size())) {
				std::cout << "Yo, max image index is "
						<< image_data.images_.size() - 1 << std::endl;
			}

			const ImageModel& im = image_data.images_[next_image_id];
			const CameraModel& cm = camera_data.cameras_[im.camera_id];

			std::string filename = argv[4] + im.image_name;

			filename.erase(std::remove(filename.begin(), filename.end(), '\n'),
					filename.end());
			filename.erase(std::remove(filename.begin(), filename.end(), '\r'),
					filename.end());

			std::cout << "Trying to open file '" << filename << "'"
					<< std::endl;
			Mat color_image = imread(filename, 1);
			if (color_image.data == NULL) {
				std::cerr << "ERROR -> Could not find file '" << filename << "'"
						<< std::endl << std::endl;
				continue;
			}

			colorPointCloud(im, cm, color_image, cloud);

			pcl::visualization::PointCloudColorHandlerRGBField<pointtype> rgb =
					pcl::visualization::PointCloudColorHandlerRGBField<pointtype>(
							cloud);
			viewer->updatePointCloud<pointtype>(cloud, rgb, "sample cloud");

			Eigen::Quaterniond quat = im.quat;
			Eigen::Translation3d trans;
			trans.x() = im.trans[0];
			trans.y() = im.trans[1];
			trans.z() = im.trans[2];

			Eigen::Transform<double, 3, Eigen::Affine> camera_to_pointcloud =
					trans * quat;
			camera_to_pointcloud = camera_to_pointcloud.inverse();

			std::vector<pcl::visualization::Camera> cameras;
			viewer->getCameras(cameras);

			pcl::visualization::Camera camera = cameras[0];

			camera.pos[0] = camera_to_pointcloud.translation().x();
			camera.pos[1] = camera_to_pointcloud.translation().y();
			camera.pos[2] = camera_to_pointcloud.translation().z();

			Eigen::Matrix3d rot = camera_to_pointcloud.affine().block<3, 3>(0,
					0);

			camera.focal[0] = rot(0, 2) + camera.pos[0];
			camera.focal[1] = rot(1, 2) + camera.pos[1];
			camera.focal[2] = rot(2, 2) + camera.pos[2];

			camera.view[0] = -rot(0, 1);
			camera.view[1] = -rot(1, 1);
			camera.view[2] = -rot(2, 1);

			viewer->setCameraParameters(camera, 0);

			imshow("StyleImage", color_image);

			waitKey(100);

		}
	}

	else {

		for (size_t i = 1; i < image_data.images_.size(); ++i) {

			int next_image_id = i;

			const ImageModel& im = image_data.images_[next_image_id];
			const CameraModel& cm = camera_data.cameras_[im.camera_id];

			std::string filename = argv[4] + im.image_name;

			filename.erase(std::remove(filename.begin(), filename.end(), '\n'),
					filename.end());
			filename.erase(std::remove(filename.begin(), filename.end(), '\r'),
					filename.end());

			std::cout << "Trying to open file '" << filename << "'"
					<< std::endl;

			Mat color_image = imread(filename, 1);
			if (color_image.data == NULL) {
				std::cerr << "ERROR -> Could not find file '" << filename << "'"
						<< std::endl << std::endl;
				continue;
			}

			colorPointCloud(im, cm, color_image, cloud);

			pcl::visualization::PointCloudColorHandlerRGBField<pointtype> rgb =
					pcl::visualization::PointCloudColorHandlerRGBField<pointtype>(
							cloud);
			viewer->updatePointCloud<pointtype>(cloud, rgb, "sample cloud");

			imshow("StyleImage", color_image);

			waitKey(100);

		}

	}

	std::cout << "Finished going through files" << std::endl;

	std::cout << "Enter a file name if you would like to save the PLY file"
			<< std::endl;
	std::string new_file;
	std::cin >> new_file;

	if (new_file.find(".ply") != std::string::npos) {

		std::cout << std::endl << "Writing to " << new_file << std::endl;
		pcl::PLYWriter writer;
		writer.write<pointtype>(new_file, *cloud, false);

	}

	std::cout << std::endl << "Ctrl-C to quit" << std::endl;
	waitKey(0);

	return 0;
}

