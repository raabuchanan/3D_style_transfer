/*
 * pcl_simple_viewer.cpp
 *
 *  Created on: Jun 11, 2017
 *      Author: perry
 */

#include <vision_project/pcl_simple_viewer.h>

namespace vision_project{

boost::shared_ptr<pcl::visualization::PCLVisualizer> createSimpleViewer(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string name){

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
			new pcl::visualization::PCLVisualizer(name));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
			cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	return viewer;

}

} // namespace vision_project

