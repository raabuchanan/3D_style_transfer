#include <memory>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>

#include <vision_project/pcl_filters.h>

namespace vision_project {

void filter_plane(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
		pcl::ModelCoefficients::Ptr coefficients,
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_out) {

	pcl::IndicesPtr indices(new std::vector<int>);

	for (int i = 0; i < cloud->points.size(); i++) {
		pcl::PointXYZRGBNormal point = cloud->points[i];
		float value = point.x * coefficients->values[0]
				+ point.y * coefficients->values[1]
				+ point.z * coefficients->values[2] + coefficients->values[3];

		if (value <= 0) {
			indices->push_back(i);
		}
	}

	pcl::ExtractIndices<pcl::PointXYZRGBNormal> eifilter;

	eifilter.setInputCloud(cloud);
	eifilter.setIndices(indices);

	eifilter.filter(*cloud_out);

}

void filter_plane(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::ModelCoefficients::Ptr coefficients,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out) {

	pcl::IndicesPtr indices(new std::vector<int>);

	for (int i = 0; i < cloud->points.size(); i++) {
		pcl::PointXYZRGB point = cloud->points[i];
		float value = point.x * coefficients->values[0]
				+ point.y * coefficients->values[1]
				+ point.z * coefficients->values[2] + coefficients->values[3];

		if (value <= 0) {
			indices->push_back(i);
		}
	}

	pcl::ExtractIndices<pcl::PointXYZRGB> eifilter;

	eifilter.setInputCloud(cloud);
	eifilter.setIndices(indices);

	eifilter.filter(*cloud_out);

}

void RadiusOutlier(pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud, double radius_search,
		int min_neighbors) {

	pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
	outrem.setInputCloud(in_cloud);
	outrem.setRadiusSearch(radius_search);
	outrem.setMinNeighborsInRadius(min_neighbors);
	outrem.filter(*out_cloud);

}

void ClusterSegmentation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloud,
		std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr >& out_clouds,
		double cluster_tolerance, int min_cluster_size, int max_cluster_size) {

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(
			new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud(in_cloud);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	ec.setClusterTolerance(cluster_tolerance); // 2cm
	ec.setMinClusterSize(min_cluster_size);
	ec.setMaxClusterSize(max_cluster_size);
	ec.setSearchMethod(tree);
	ec.setInputCloud(in_cloud);
	ec.extract(cluster_indices);

	for (int i = 0; i < cluster_indices.size(); ++i) {
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(
				new pcl::PointCloud<pcl::PointXYZRGB>);
		for (std::vector<int>::const_iterator pit =
				cluster_indices[0].indices.begin();
				pit != cluster_indices[0].indices.end(); ++pit)
			cloud_cluster->points.push_back(in_cloud->points[*pit]); //*
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		out_clouds.push_back(cloud_cluster);
	}
}

} // namespace vision_project
