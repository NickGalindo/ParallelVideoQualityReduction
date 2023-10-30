#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <typeinfo>

#define ff first
#define ss second

using namespace std;
typedef long long ll;

int output_width, output_height, input_width, input_height, input_channels, threads;
double fps, scale_factor;

// Special class to manage img data manually
class SizedImg{
public:
  int height, width, channels;
  // REMEMBER IT IS ROW FIRST THEN COLUMN IE pixelData[height][width][channel]
  unsigned char *img;

  SizedImg(int h, int w, int c, unsigned char* i){
    height = h;
    width = w;
    channels = c;
    img = i;
  }
};

void frameReduction(SizedImg &out, SizedImg &in);
pair<cv::VideoCapture, cv::VideoWriter>  validateInput(int argc, char* argv[]);
SizedImg parallelResizeImg(SizedImg img);



int main(int argc, char* argv[]){
  // Fast io
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  pair<cv::VideoCapture, cv::VideoWriter> inout_videos = validateInput(argc, argv);


  // Iterate thorugh the frames of the video
  cv::Mat frame;
  while(inout_videos.ff.read(frame)){
    input_channels = frame.channels();

    SizedImg img(input_height, input_width, input_channels, (unsigned char*)frame.data);
    SizedImg new_img = parallelResizeImg(img);

    cv::Mat new_frame(output_height, output_width, CV_8UC3, new_img.img);
    inout_videos.ss.write(new_frame);

    free(new_img.img);
  }

  // Release both files
  inout_videos.ff.release();
  inout_videos.ss.release();

  cout << "Video processing completed." << endl;

  return 0;
}


// Frame reduction function for synchornization on thread ending
void frameReduction(SizedImg &out, SizedImg &in){
  for(int i = 0; i < out.height; i++)
    for(int j = 0; j < out.width; j++)
      for(int k = 0; k < out.channels; k++)
        out.img[i*out.width*out.channels + j*out.channels + k] = in.img[i*out.width*out.channels + j*out.channels + k];
}


pair<cv::VideoCapture, cv::VideoWriter>  validateInput(int argc, char* argv[]){
  // Check if number of arguments matches
  if(argc != 4){
    cerr << "INPUT ERROR: Usage -> video_reduction input_video_path.mpg output_video_path.mpg #ofThreads" << endl;
    exit(1);
  }

  // Get paths of input videos and output videos
  string input_video_path = argv[1];
  string output_video_path = argv[2];

  // Turn argument threads into numbers
  try{
    threads = stoi(argv[3]);
  }catch(const invalid_argument& e){
    cerr << "INPUT ERROR: " << e.what() << endl;
    exit(1);
  }catch(const out_of_range& e){
    cerr << "INPUT ERROR: " << e.what() << endl;
    exit(1);
  }

  // Recieve input video
  cv::VideoCapture inputVideo(input_video_path);
  if(!inputVideo.isOpened()){
    cerr << "ERROR: Couldn't open input video" << endl;
    exit(1);
  }

  // Calculate video properties
  input_width = (int) inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
  input_height = (int) inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);

  if(input_height == 1080){
    output_height = 360;
    scale_factor = (double) (input_height) / (double) (output_height);
    output_width = input_width / scale_factor;
  }else{
    output_width = 360;
    scale_factor = (double) (input_width) / (double) (output_width);
    output_height = input_height / scale_factor;
  }

  // Create output video
  fps = inputVideo.get(cv::CAP_PROP_FPS);
  cv::VideoWriter outputVideo(output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(output_width, output_height), true);
  if(!inputVideo.isOpened()){
    cerr << "ERROR: Couldn't create output video" << endl;
    exit(1);
  }

  return {inputVideo, outputVideo};
}

SizedImg parallelResizeImg(SizedImg img){
  #pragma omp declare reduction(frameReduction : SizedImg : frameReduction(omp_out, omp_in)) initializer (omp_priv=omp_orig)

  SizedImg new_img(output_height, output_width, input_channels, (unsigned char*) malloc(output_height * output_width * input_channels * sizeof(unsigned char)));

  for(int i = 0; i < new_img.height; i++)
    for(int j = 0; j < new_img.width; j++)
      for(int k = 0; k < new_img.channels; k++)
        new_img.img[i*output_width*input_channels + j*input_channels + k] = 0;

  #pragma omp parallel for num_threads(threads) firstprivate(output_height, output_width, input_height, input_width, input_channels, scale_factor) reduction(frameReduction : new_img)
  for(int i = 0; i < output_height; i++){
    for(int j = 0; j < output_width; j++){
      for(int k = 0; k < input_channels; k++){
        unsigned int avg = 0;
        for(int x = i*scale_factor; x < (i+1)+scale_factor; x++)
          for(int y = j*scale_factor; y < (j+1)+scale_factor; y++)
            avg += (unsigned int) img.img[x*input_width*input_channels + y*input_channels + k];
  
        avg = avg/(scale_factor*scale_factor);
  
        new_img.img[i*output_width*input_channels + j*input_channels + k] = (unsigned char) avg;
      }
    }
  }

  return new_img;
}
