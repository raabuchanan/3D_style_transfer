#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string> 
#include <iomanip>
#include "opencv2/xfeatures2d.hpp"
#include <fstream>
#include <sqlite3.h>
#include <vector>
#include <unistd.h>

// 5000 features is needed for proper 3D reconstruction but this requires 
// a significant amount of memory. Also all images should have features extracted.
#define NUM_KEYPOINTS 3000
#define NUM_IMAGES_TO_EXTRACT 5

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


void readme();

int main( int argc, char** argv )
{
	String folderpath = "../data/glove/mosaic/*.png";
	vector<String> filenames;
	glob(folderpath, filenames);
	const char *dbFilename = "glove_mosaic_learned.db";
	sqlite3 *db;
    char * sql;
    sqlite3_stmt * stmt = NULL;
    float keypointBuffer[NUM_KEYPOINTS][4];
    char* descriptorBuffer = new char[NUM_KEYPOINTS*4096];
    int image_id = 1;

    CALL_SQLITE (open (dbFilename, &db));

	for (int i=0; i<NUM_IMAGES_TO_EXTRACT; i++)
	{
	    Mat img = imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE );

    	if( !img.data)
		{ 
			printf(" --(!) Error reading images \n");
			continue;
		}

		printf("Processing: %s\n", filenames[i].c_str());

		// COLAMP uses contrastThreshold=0.0067 which produces many low quality keypoints
		cv::Ptr<Feature2D> detector = xfeatures2d::SIFT::create(NUM_KEYPOINTS,3,0.01,10,1.6);

		vector<KeyPoint> keypoints;
		detector->detect(img, keypoints);

		printf("%ld Keypoints found\n", keypoints.size());

		if(keypoints.size() > NUM_KEYPOINTS){
			keypoints.resize(NUM_KEYPOINTS);
		}
		cv::Mat outImage;


		for(int j=0; j<keypoints.size();j++)
		{
			int desc_it = 0;
			for(int n=round(keypoints[j].pt.y);n<round(keypoints[j].pt.y) + 64; n++)
			{
				for(int m=round(keypoints[j].pt.x);m<round(keypoints[j].pt.x) + 64; m++)
				{

					descriptorBuffer[4096*j + desc_it] = img.at<uchar>(n,m);
					desc_it++;
				}				
			}

			keypointBuffer[j][0] = keypoints[j].pt.x;
			keypointBuffer[j][1] = keypoints[j].pt.y;
			keypointBuffer[j][2] = keypoints[j].size;
			keypointBuffer[j][3] = keypoints[j].angle;
		}

		int num_keys = (int)keypoints.size();

		// Saving Keypoints to database
	    CALL_SQLITE (prepare_v2 (db, "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, '4', ? )", -1, &stmt, NULL));
	    CALL_SQLITE (bind_int(stmt, 1, image_id));
	    CALL_SQLITE (bind_int(stmt, 2, num_keys));
	    CALL_SQLITE (bind_blob(stmt, 3, keypointBuffer,num_keys*4*sizeof(float),SQLITE_STATIC));

	    CALL_SQLITE_EXPECT (step (stmt), DONE);
	    printf ("Wrote data to row id %d\n", (int) sqlite3_last_insert_rowid (db));

	    // Saving 64x64 patches to database
	    CALL_SQLITE (prepare_v2 (db, "INSERT INTO descriptors (image_id, rows, cols, data) VALUES (?, ?, '4096', ? )", -1, &stmt, NULL));
	    CALL_SQLITE (bind_int(stmt, 1, image_id));
	    CALL_SQLITE (bind_int(stmt, 2, num_keys));
	    CALL_SQLITE (bind_blob(stmt, 3, descriptorBuffer,num_keys*64*64*sizeof(char),SQLITE_STATIC));

	    CALL_SQLITE_EXPECT (step (stmt), DONE);
	    printf ("Wrote data to row id %d\n", (int) sqlite3_last_insert_rowid (db));


	    sqlite3_finalize(stmt);
	    image_id++;
	}

	sqlite3_close(db);

	delete[] descriptorBuffer;
}
