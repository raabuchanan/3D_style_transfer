
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <vision_project/pcl_simple_viewer.h>

typedef pcl::PointXYZRGB pointtype;

int
 main (int argc, char** argv)
{  
  pcl::PointCloud<pointtype>::Ptr cloud (new pcl::PointCloud<pointtype>);

  if (argc <= 1){
    PCL_ERROR ( "Need a file as an input");
    return (-1);
  }

  if (pcl::io::loadPLYFile(argv[1], *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file \n");
    return (-1);
  }


  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = vision_project::createSimpleViewer(cloud, "viewer");
  viewer->setBackgroundColor (0.1, 0.1, 0.1);

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

  return 0;
}
