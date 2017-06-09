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

#define NUM_KEYPOINTS 3000

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

	//const char *dbFilename = "style_cup_194.db";
	const char *dbFilename = "glove_mosaic_learned.db";
	sqlite3 *db;
    char * sql;
    sqlite3_stmt * stmt = NULL;
    float keypointBuffer[NUM_KEYPOINTS][4];
    char* descriptorBuffer = new char[NUM_KEYPOINTS*4096];
    int image_id = 1;

    CALL_SQLITE (open (dbFilename, &db));

	for (int i=0; i<filenames.size(); i++)
	{
	    Mat img = imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE );

    	if( !img.data)
		{ 
			printf(" --(!) Error reading images \n");
			continue;
		}

		printf("Processing: %s\n", filenames[i].c_str());

		cv::Ptr<Feature2D> detector = xfeatures2d::SIFT::create(NUM_KEYPOINTS,3,0.0067,10,1.6);

		vector<KeyPoint> keypoints;
		detector->detect(img, keypoints);

		printf("%ld Keypoints found\n", keypoints.size());

		if(keypoints.size() > NUM_KEYPOINTS){
			keypoints.resize(NUM_KEYPOINTS);
		}


		cv::Mat outImage;

		float currentSize = keypoints[0].size;
		float scale = 1.0;
		Mat image((int)round(img.rows*scale),(int)round(img.cols*scale),CV_8U);

		resize(img,image,image.size(),0,0,INTER_NEAREST);


		//Mat paddedImage(image.rows + 64,image.cols + 64,CV_8U);
		//copyMakeBorder( image, paddedImage, 32, 32, 32, 32, BORDER_CONSTANT, 0 );

		for(int j=0; j<keypoints.size();j++)
		{

			//cout << "X: " << keypoints[j].pt.x << " Y: " << keypoints[j].pt.y << " size: " << keypoints[j].size <<  " angle: " << keypoints[j].angle << endl;

			// if(keypoints[j].size > currentSize)
			// {
			// 	currentSize = keypoints[j].size;
			// 	scale = scale / 1.1;
			// 	Mat image((int)round(img.rows*scale),(int)round(img.cols*scale),CV_8U);
			// 	resize(img,image,image.size(),0,0,INTER_NEAREST);
			// }


			int desc_it = 0;
			for(int n=round(scale*keypoints[j].pt.y);n<round(scale*keypoints[j].pt.y) + 64; n++)
			{
				for(int m=round(scale*keypoints[j].pt.x);m<round(scale*keypoints[j].pt.x) + 64; m++)
				{

					descriptorBuffer[4096*j + desc_it] = image.at<uchar>(n,m);
					desc_it++;
				}				
			}

			// float tempScale = 1/scale;

			keypointBuffer[j][0] = keypoints[j].pt.x;
			keypointBuffer[j][1] = keypoints[j].pt.y;
			keypointBuffer[j][2] = keypoints[j].size;//1/scale;
			keypointBuffer[j][3] = keypoints[j].angle;

			// memcpy(&keypointBuffer[j][0], &keypoints[j].pt.x, sizeof(float));
			// memcpy(&keypointBuffer[j][4], &keypoints[j].pt.y, sizeof(float));
			// memcpy(&keypointBuffer[j][8], &tempScale, sizeof(float));
			// memcpy(&keypointBuffer[j][12], &keypoints[j].angle, sizeof(float));

		}


/*
	    CALL_SQLITE (prepare_v2 (finaldb, "INSERT INTO keypoints (image_id, rows,cols,data) VALUES (?, '2000', '4', ? )", -1, &stmt, NULL));
	    CALL_SQLITE (bind_int(stmt, 1, image_id));
	    CALL_SQLITE (bind_blob(stmt, 2, keypointBuffer,sizeof(keypointBuffer),SQLITE_STATIC));

	    */

/*
	    int rc = sqlite3_prepare_v2 (finaldb, "INSERT INTO keypoints (image_id, rows,cols,data) VALUES (?, '2000', '4', ?);", -1, &stmt, NULL);

        if (rc != SQLITE_OK)
	    	cout << "prepare failed: " << sqlite3_errmsg(finaldb) << endl;

	    rc = sqlite3_bind_int(stmt, 1, i + 1);

        if (rc != SQLITE_OK)
	    	cout << "bind in failed: " << sqlite3_errmsg(finaldb) << endl;

	    rc = sqlite3_bind_blob(stmt, 2, keypointBuffer,sizeof(keypointBuffer),SQLITE_STATIC);

        if (rc != SQLITE_OK)
	    	cout << "bind blob failed: " << sqlite3_errmsg(finaldb) << endl;


	    rc = sqlite3_step (stmt);
	    printf ("Wrote data to row id %d\n", (int) sqlite3_last_insert_rowid (finaldb));

        if (rc != SQLITE_DONE)
	    	cout << "step failed: " << rc << endl;
*/

		int num_keys = (int)keypoints.size();

	    CALL_SQLITE (prepare_v2 (db, "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, '4', ? )", -1, &stmt, NULL));
	    CALL_SQLITE (bind_int(stmt, 1, image_id));
	    CALL_SQLITE (bind_int(stmt, 2, num_keys));
	    CALL_SQLITE (bind_blob(stmt, 3, keypointBuffer,num_keys*4*sizeof(float),SQLITE_STATIC));

	    CALL_SQLITE_EXPECT (step (stmt), DONE);
	    printf ("Wrote data to row id %d\n", (int) sqlite3_last_insert_rowid (db));

	    CALL_SQLITE (prepare_v2 (db, "INSERT INTO descriptors (image_id, rows, cols, data) VALUES (?, ?, '4096', ? )", -1, &stmt, NULL));
	    CALL_SQLITE (bind_int(stmt, 1, image_id));
	    CALL_SQLITE (bind_int(stmt, 2, num_keys));
	    CALL_SQLITE (bind_blob(stmt, 3, descriptorBuffer,num_keys*64*64*sizeof(char),SQLITE_STATIC));

	    CALL_SQLITE_EXPECT (step (stmt), DONE);
	    printf ("Wrote data to row id %d\n", (int) sqlite3_last_insert_rowid (db));


	    sqlite3_finalize(stmt);
	    image_id++;

	    // if(i>9){
	    // 	break;
	    // }

	}


	//sqlite3_close(finaldb);
	sqlite3_close(db);

	delete[] descriptorBuffer;
}
