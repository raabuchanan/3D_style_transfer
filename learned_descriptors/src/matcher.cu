#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string> 
#include <iomanip>
//#include "opencv2/line_descriptor/descriptor.hpp"
#include <fstream>
#include <sqlite3.h>
#include <vector>
#include <unistd.h>
#include <algorithm>
#include <Eigen/Dense>
#include <iterator>

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


#define MINIMUM_THRESH 1.25

using namespace cv;
using namespace std;

vector<Eigen::MatrixXf> database;
int image_count;

// Callback for loading descriptors from database
static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
	float tempFloat;
	const char *buffer = argv[3];

	int numRows = atoi(argv[1]);

	int image_id = atoi(argv[0]);
	Eigen::MatrixXf descriptors = Eigen::MatrixXf::Zero(numRows, 128);

	for (int i = 0; i < numRows; i++) 
	{
		for (int j = 0; j < 128; j++) {
		  memcpy(&tempFloat, buffer + 4 * j + (4 * i * 128), sizeof(float));
		  descriptors(i, j) = tempFloat;
		}
	}

	cout << "Loading Image: " << image_id << endl;
	image_count++;
	database.push_back(descriptors.transpose()); // .data() is returned as column major

	return 0;
}

sqlite3_int64 getPairID(sqlite3_int64 id1, sqlite3_int64 id2) {
	sqlite3_int64 pairID;
  if (id1 > id2) {
    pairID = 2147483647 * id2 + id1;
  } else {
    pairID = 2147483647 * id1 + id2;
  }

  return pairID;
}

void getImageIDs(long pairID, int &id1, int &id2) {
  id2 = pairID % 2147483647;
  id1 = (pairID - id2) / 2147483647;
}

__global__  void cuFindMatches(float* im1, float* im2, int* matches, float* SSDs, int rows1, int rows2) {
	float ssd;
	float diff;
	__shared__ float best;
	__shared__ int bestIndx;

    if (threadIdx.x == 0) {

      best = MINIMUM_THRESH;
      bestIndx = -1;
    }

    __syncthreads();

	int idx = threadIdx.x;
	int stride = blockDim.x;

	// Get initial matches
	for(int ii = idx; ii<rows2; ii +=stride){
		ssd = 0;
		for(int k = 0; k<128; k++){
			diff = im1[k + 128*blockIdx.x] - im2[k + 128*ii];
			ssd += diff*diff;
		}

		ssd = sqrt(ssd);

		if(ssd < best){
			best = ssd;
			bestIndx = ii;
		}
	}


	matches[blockIdx.x] = bestIndx;
	SSDs[blockIdx.x] = best;

	__syncthreads();
	if(matches[blockIdx.x] >= 0){
		__syncthreads();
		if(matches[blockIdx.x] >= 0){
			// Remove duplicates and non mutual matches
			for(int jj = idx; jj<rows1; jj += stride){
				if(blockIdx.x != jj){
					if(matches[blockIdx.x] == matches[jj]){
						matches[blockIdx.x] = -1;
						matches[jj] = -1;
					}
				}
			}
		}
	}


}


int main( int argc, char** argv )
{
	char *zErrMsg = 0;
	//const char *dbFilename = "style_cup_194.db";
	const char *dbFilename = "glove_mosaic_learned.db";
	sqlite3 *db;
	char * sql;

	image_count = 0;

	// Read descriptors from database
	CALL_SQLITE (open (dbFilename, &db));
	sql = "SELECT * from learned";
	CALL_SQLITE (exec(db, sql, callback, 0, &zErrMsg));


	float *d_im1, *d_im2, *d_SSDs;
	int *d_matches;
    float *im1;
    float *im2;
    sqlite3_stmt * stmt = NULL;
	float *SSDs1;
	int *matches1;
	int *output_buffer;
	int rows1, rows2;

    vector<int> blobData;
    sqlite3_int64 pairID;

    int config = 0;

    for(int i=0; i<image_count;i++)
    {
    	im1 = database[i].data();
    	rows1 = database[i].cols();
    	cudaMalloc((void **)&d_im1,rows1*128*4);
    	cudaMemcpy(d_im1, im1, rows1*128*4, cudaMemcpyHostToDevice);

		cudaMalloc((void **)&d_matches,rows1*4);
		cudaMalloc((void **)&d_SSDs,rows1*4);

	    SSDs1 = new float[rows1];
	    matches1 = new int[rows1];
  
		for(int j = i; j<image_count; j++)
		{
			if((i!=j) && (j < i + 30)){ // Only comparing images within 30 frames

		    	im2 = database[j].data();
		    	rows2 = database[j].cols();
		    	cudaMalloc((void **)&d_im2,rows2*128*4);
	    		cudaMemcpy(d_im2, im2, rows2*128*4, cudaMemcpyHostToDevice);

				cout << "Processing Image " << i + 1 << " and Image " << j + 1 << endl;

				cudaMemset(d_SSDs, MINIMUM_THRESH, rows1*4);

			    cuFindMatches<<<rows1, 1024>>>(d_im1,d_im2,d_matches,d_SSDs,rows1,rows2);

			    cudaMemcpy(matches1, d_matches, rows1*sizeof(int), cudaMemcpyDeviceToHost);
			    cudaMemcpy(SSDs1, d_SSDs, rows1*sizeof(int), cudaMemcpyDeviceToHost);


			    for(int k = 0; k<rows1; k++)
			    {
			    	//printf("%d matches1 %d ssd1 %f matches2 %d ssd2 %f\n", k, matches1[k], SSDs1[k], matches2[k], SSDs2[k]);
			    	if(matches1[k]>=0)
			    	{
						blobData.push_back(k);
						blobData.push_back(matches1[k]);
				    }
			    }

			    cout << "[VERIFIED] " << blobData.size()/2 << " Matches " << endl;

			    output_buffer = new int[blobData.size()];

			    std::copy(blobData.begin(), blobData.end(), output_buffer);

			    //cout << blobData.size()*2*4 << endl;
			    config = 0;
			    if(blobData.size()/2 > 0){
			    	if(blobData.size()/2 > 100){
			    		config = 6;
			    	}else{
			    		config = 3;
			    	}
			    }

			    // Save matches to database, COLMAP only actually looks at inlier_matches
			    pairID = getPairID((sqlite3_int64)(i+1), (sqlite3_int64)(j+1));

			    CALL_SQLITE (prepare_v2 (db, "INSERT INTO matches (pair_id, rows, cols, data) VALUES (?, ?, '2', ? )", -1, &stmt, NULL));
			    CALL_SQLITE (bind_int64(stmt, 1, pairID));
			    CALL_SQLITE (bind_int(stmt, 2, blobData.size()/2));
			    CALL_SQLITE (bind_blob(stmt, 3, output_buffer,blobData.size()*4,SQLITE_STATIC));

			    CALL_SQLITE_EXPECT (step (stmt), DONE);
			    printf ("Wrote data to row id %lld\n", sqlite3_last_insert_rowid (db));

			    CALL_SQLITE (prepare_v2 (db, "INSERT INTO inlier_matches (pair_id, rows, cols, data, config) VALUES (?, ?, '2', ?, ? )", -1, &stmt, NULL));
			    CALL_SQLITE (bind_int64(stmt, 1, pairID));
			    CALL_SQLITE (bind_int(stmt, 2, blobData.size()/2));
			    CALL_SQLITE (bind_blob(stmt, 3, output_buffer,blobData.size()*4,SQLITE_STATIC));
			    CALL_SQLITE (bind_int(stmt, 4, config));

			    CALL_SQLITE_EXPECT (step (stmt), DONE);
			    printf ("Wrote data to row id %lld\n", sqlite3_last_insert_rowid (db));

			    sqlite3_finalize(stmt);

			    blobData.clear();

				cudaFree(d_im2);

				delete[] output_buffer;
			}

		}

		delete[] SSDs1;
		delete[] matches1;

		cudaFree(d_im1);
		cudaFree(d_matches);
		cudaFree(d_SSDs);
	}

	cudaFree(d_im1);
	cudaFree(d_im2);
	cudaFree(d_matches);
	cudaFree(d_SSDs);


	CALL_SQLITE (close(db));
}
