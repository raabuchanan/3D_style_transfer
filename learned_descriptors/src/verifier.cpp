#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string> 
#include <iomanip>
#include <fstream>
#include <sqlite3.h>
#include <vector>
#include <unistd.h>
#include <algorithm>

#define CALL_SQLITE(f)                                          \
    {                                                           \
        int i;                                                  \
        i = sqlite3_ ## f;                                      \
        if (i != SQLITE_OK) {                                   \
            fprintf (stderr, "%s failed with status %d: %s\n",  \
                     #f, i, sqlite3_errmsg (db));               \
            exit (1);                                           \
        }                                                       \
    }                                                           \

#define CALL_SQLITE_EXPECT(f,x)                                 \
    {                                                           \
        int i;                                                  \
        i = sqlite3_ ## f;                                      \
        if (i != SQLITE_ ## x) {                                \
            fprintf (stderr, "%s failed with status %d: %s\n",  \
                     #f, i, sqlite3_errmsg (db));               \
            exit (1);                                           \
        }                                                       \
    }                                                           \


using namespace cv;
using namespace std;


vector<KeyPoint> keypoints1;
vector<KeyPoint> keypoints2;

vector<DMatch> matches1to2;

static int matchesCallback(void *NotUsed, int argc, char **argv, char **azColName) {
  long pairID = atol(argv[0]);


  if(pairID == 2147483649){
    long rows = atoi(argv[1]);
    long cols = atoi(argv[2]);
    const char *buffer = argv[3];
    int query, train;



    for (int i = 0; i < rows; i++) {
        memcpy(&query, buffer + (4 * i * 2), sizeof(int));
        memcpy(&train, buffer + 4 + (4 * i * 2), sizeof(int));
        cv::DMatch match(query,train,-1);
        matches1to2.push_back(match);
    }
  }

  return 0;
}


static int keypointCallback(void *NotUsed, int argc, char **argv, char **azColName) {
  int image_id = atoi(argv[0]);

  if(image_id == 1 || image_id == 2){

    long rows = atoi(argv[1]);
    long cols = atoi(argv[2]);
    const char *buffer = argv[3];
    float x,y,size,angle;

    for (int i = 0; i < rows; i++)
    {
        memcpy(&x, buffer + (4 * i * 4), sizeof(float));

        memcpy(&y, buffer + 4 + (4 * i * 4), sizeof(float));

        memcpy(&size, buffer + 8 + (4 * i * 4), sizeof(float));

        memcpy(&angle, buffer + 12 + (4 * i * 4), sizeof(float));

        cv::Point2f pt_(x, y);
        cv::KeyPoint kpt_(pt_, size, angle);

        if (image_id == 1)
        {
            keypoints1.push_back(kpt_);

            
        }
        else
        {
            keypoints2.push_back(kpt_);

            //cout << "i: " << i <<  "X: " << x << " Y: " << y << " Size: " << size << " Angle " << angle << endl;
        }

        
     }
  }

  return 0;
}

void getImageIDs(sqlite3_int64 pairID, int &id1, int &id2) {
  id2 = pairID % 2147483647;
  id1 = (pairID - id2) / 2147483647;
}


int main( int argc, char** argv )
{
  char *zErrMsg = 0;
  //const char *dbFilename = "style_cup_194.db";
  const char *dbFilename = "glove_mosaic_learned.db";
  sqlite3 *db;
  char * sql;

  CALL_SQLITE (open (dbFilename, &db));

  sql = "SELECT * from inlier_matches";
  CALL_SQLITE (exec(db, sql, matchesCallback, 0, &zErrMsg));

  sql = "SELECT * from keypoints";
  CALL_SQLITE (exec(db, sql, keypointCallback, 0, &zErrMsg));

  String folderpath = "/home/raab/3D_model_style_transfer/data/glove/mosaic/*.png";
  vector<String> filenames;
  glob(folderpath, filenames);

  Mat img1 = imread(filenames[0], CV_LOAD_IMAGE_GRAYSCALE );

  Mat img2 = imread(filenames[1], CV_LOAD_IMAGE_GRAYSCALE );

  cv::Mat outImage1, outImage2, outImage3;

  drawKeypoints(img1,keypoints1,outImage1);
  drawKeypoints(img2,keypoints2,outImage2);

  drawMatches(img1,keypoints1,img2,keypoints2,matches1to2,outImage3);

  namedWindow( "1 window", WINDOW_AUTOSIZE );// Create a window for original.
  imshow( "1 window", outImage1 );                   // Show our image inside it.

    namedWindow( "2 window", WINDOW_AUTOSIZE );// Create a window for original.
  imshow( "2 window", outImage2 );                   // Show our image inside it.

    namedWindow( "3 window", WINDOW_AUTOSIZE );// Create a window for original.
  imshow( "3 window", outImage3 );                   // Show our image inside it.


  // Loop until escape is pressed
  while (cvWaitKey(0) != '\33') {

  }

  CALL_SQLITE (close(db));
}
