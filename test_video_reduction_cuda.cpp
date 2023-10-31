#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <typeinfo>

#define ff first
#define ss second

using namespace std;
typedef long long ll;

string inputVideoName = "video.mp4"

    __global__ void
    compressFrames(unsigned char **batchFrames, unsigned char **compressedFrames, int batchSize, int frameWidth, int frameHeight, int compressedFrameWidth, int compressedFrameHeight)
{

  int frameId = blockIdx.x;
  int frameSize = frameWidth * frameHeight * 3;
  int compressedFrameSize = compressedFrameWidth * compressedFrameHeight * 3;
  int frameOffset = frameId * frameSize;
  int compressedFrameOffset = frameId * compressedFrameSize;

  // Compress frame
  for (int i = 0; i < compressedFrameHeight; i++)
  {
    for (int j = 0; j < compressedFrameWidth; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        //  TODO fill each pixel of the compressed frame with the average of neighboring pixels
        // The data for each frame starts at *(compressedFrames + blockIdx.x + (compressedFrameWidth ))
        unsigned char *targetPixel = compressedFrames + (i * compressedFrameWidth + j) * 3 + k;
        // sum the 9 pixels around the target pixel
        int sum = 0;
        for (int ii = 0; ii < 3; ii++)
        {
          for (int jj = 0; jj < 3; jj++)
          {
            sum += *(batchFrames + (i * 3 + ii) * frameWidth + (j * 3 + jj) * 3 + k);
          }
        }
        *targetPixel = sum / 9;
      }
    }
  }
}

int main()
{
  cv::VideoCapture inputVideo(inputVideoName);
  int frameWidth = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
  int frameHeight = inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
  int numFrames = inputVideo.get(cv::CAP_PROP_FRAME_COUNT);
  int fps = inputVideo.get(cv::CAP_PROP_FPS);
  int compressedFrameWidth = frameWidth / 3;
  int compressedFrameHeight = frameHeight / 3;
  cv::VideoWriter outputVideo(
      "nuevo_video.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps,
      cv::Size(compressedFrameWidth, compressedFrameHeight), true);

  cv::Mat frame;
  // use malloc memory to store batchSize integers

  // malloc memory to store a pointer to each frame.data in the batch
  unsigned char **batchFrames = (unsigned char **)malloc(batchSize * sizeof(unsigned char *));
  // Reserve that same memory in the gpu
  unsigned char **d_batchFrames;
  cudaMalloc((void **)&d_batchFrames, batchSize * sizeof(unsigned char *));
  // Reserve memory for the compressed frames
  unsigned char **compressedFrames = (unsigned char **)malloc(batchSize * sizeof(unsigned char *));
  // Reserve that same memory in the gpu
  unsigned char **d_compressedFrames;
  cudaMalloc((void **)&d_compressedFrames, batchSize * sizeof(unsigned char *));

  // Process batches of batchSize frames, keep passing data to gpu and writing to output
  while (true)
  {
    cv::Mat frame;
    inputVideo >> frame;

    if (frame.empty())
      break;

    batchFrames[batchCnt] = frame.data;
    frames++;
    batchCnt++;

    if (batchCnt == batchSize)
    {

      // Copy batchFrames to gpu
      cudaMemcpy(d_batchFrames, batchFrames, batchSize * sizeof(unsigned char *), cudaMemcpyHostToDevice);
      // Compress batchFrames
      compressFrames<<<batchSize, 2048>>>(d_batchFrames, d_compressedFrames, batchSize, frameWidth, frameHeight, compressedFrameWidth, compressedFrameHeight);
      // Write to output
      batchCnt = 0;
    }
  }

  return 0;
}