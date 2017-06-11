// filter_point_cloud.cpp
//     Author: Perry Franklin
//
// This file provides a simple way to filter a point cloud.
// Unfortunately, it does not provide run-time changes, so
// changes to the filters needs to be done in this file and
// then compiled into the executable.
//
// Currently, it assumes there is a ground plane that can
// be extracted and filtered out. Various other filters are
// applied to extract the object of interest (hopefully).

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <vision_project/pcl_filters.h>
#include <vision_project/pcl_simple_viewer.h>

typedef pcl::PointXYZRGB pointtype;

int main(int argc, char** argv) {

	std::cerr << "Starting" << std::endl;

	if (argc <= 2) {
		PCL_ERROR("Need a file as an input and output location");
		PCL_ERROR("./filter_point_cloud FILE_TO_READ FILE_TO_WRITE");
		return (-1);
	}
	pcl::PointCloud<pointtype>::Ptr cloud(new pcl::PointCloud<pointtype>);

	std::cout << "hello" << cloud << std::endl;

	if (pcl::io::loadPLYFile(argv[1], *cloud) == -1) //* load the file
			{
		PCL_ERROR("Couldn't read file \n");
		return (-1);
	}

	std::cout << "Loaded " << argv[1] << std::endl;

// PLANE FILTER =============================

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pointtype> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.01);

	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);

	if (inliers->indices.size() == 0) {
		PCL_ERROR("Could not estimate a planar model for the given dataset.");
		return (-1);
	}

	std::cerr << "Model coefficients: " << coefficients->values[0] << " "
			<< coefficients->values[1] << " " << coefficients->values[2] << " "
			<< coefficients->values[3] << std::endl;

	pcl::PointCloud<pointtype>::Ptr new_cloud(new pcl::PointCloud<pointtype>);

	coefficients->values[3] += 0.1;

	vision_project::filter_plane(cloud, coefficients, new_cloud);

	std::cout<<"Removed plane (hopefully) -- if it seem like the plane filter is removing the cloud you want, it may be removing the wrong half-plane"<<std::endl;

// ============================================================

	pcl::PointCloud<pointtype>::Ptr cloud_filtered(
			new pcl::PointCloud<pointtype>);
	vision_project::RadiusOutlier(new_cloud, cloud_filtered, 0.05, 5);

	std::vector<pcl::PointCloud<pointtype>::Ptr> clusters;

	vision_project::ClusterSegmentation(cloud_filtered,
			clusters,
			0.02, 10000, 2500000);

	std::cout<<"Segmented out the clusters"<<std::endl;

	pcl::PointCloud<pointtype>::Ptr cloud_cluster = clusters[0];

	pcl::PointCloud<pointtype>::Ptr new_cloud_cluster(
			new pcl::PointCloud<pointtype>);

	vision_project::RadiusOutlier(cloud_cluster, new_cloud_cluster, 0.1, 20);

	pcl::visualization::PointCloudColorHandlerCustom<pointtype> single_color(
			cloud_cluster, 0, 255, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pointtype> rgb2(
			cloud_cluster);


	std::cout<<"Writing ply file '"<<argv[2]<<"'"<<std::endl;

	pcl::PLYWriter writer;
	writer.write<pointtype>(argv[2], *new_cloud_cluster, false); //*

	std::cout<<"Done"<<std::endl;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = vision_project::createSimpleViewer(new_cloud, "Removed Plane");

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2 = vision_project::createSimpleViewer(cloud_filtered, "Outlier Filtering");

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer3 = vision_project::createSimpleViewer(cloud_cluster, "Largest Cluster");

	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
		viewer2->spinOnce(100);
		viewer3->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return (0);
}
