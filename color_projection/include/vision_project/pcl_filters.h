#ifndef VISION_PROJECT_PLANE_FILTER_H
#define VISION_PROJECT_PLANE_FILTER_H

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace vision_project {

void filter_plane(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
		pcl::ModelCoefficients::Ptr coefficients,
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_out);

void filter_plane(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::ModelCoefficients::Ptr coefficients,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out);

void RadiusOutlier(pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud, double radius_search,
		int min_neighbors);

void ClusterSegmentation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloud,
		std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr >& out_clouds,
		double cluster_tolerance, int min_cluster_size, int max_cluster_size);

} // namespace vision_project

#endif /* VISION_PROJECT_PLANE_FILTER_H */
