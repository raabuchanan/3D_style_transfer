/*
 * pcl_simple_viewer.h
 *
 *  Created on: Jun 11, 2017
 *      Author: perry
 */

#ifndef INCLUDE_VISION_PROJECT_PCL_SIMPLE_VIEWER_H_
#define INCLUDE_VISION_PROJECT_PCL_SIMPLE_VIEWER_H_

#include <pcl/visualization/pcl_visualizer.h>

namespace vision_project{

boost::shared_ptr<pcl::visualization::PCLVisualizer> createSimpleViewer(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string name);

}


#endif /* INCLUDE_VISION_PROJECT_PCL_SIMPLE_VIEWER_H_ */
