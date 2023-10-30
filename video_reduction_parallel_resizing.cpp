#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <typeinfo>

#define ff first
#define ss second

using namespace std;
typedef long long ll;

// CLASS DEFINITIONS
class SizedImg{
public:
  int height=0, width=0, channels=0;
  // REMEMBER IT IS ROW FIRST THEN COLUMN IE pixelData[height][width][channel]
  unsigned char* img;

  SizedImg(int h, int w, int c, unsigned char* i){
    height = h;
    width = w;
    channels = c;
    img = i;
  }
  SizedImg(){}
};

// VARIABLE DEFINITIONS
int output_width, output_height, input_width, input_height, input_channels, threads, frames;
double fps, scale_factor;

// FUNCTION DEFINITIONS
void framesReduction(vector<SizedImg> &out, vector<SizedImg> &in);
pair<cv::VideoCapture, cv::VideoWriter>  validateInput(int argc, char* argv[]);
SizedImg resizeImg(SizedImg &img);


// MAIN
int main(int argc, char* argv[]){
  // Fast io
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  pair<cv::VideoCapture, cv::VideoWriter> inout_videos = validateInput(argc, argv);

  // Extract all frames from video
  vector<cv::Mat> vid_data;
  frames = 0;
  while(true){
    cv::Mat frame;
    inout_videos.ff >> frame;
    
    if(frame.empty()) break;

    input_channels = frame.channels();
    vid_data.push_back(frame);
    frames++;
  }

  vector<SizedImg> final_vid;
  final_vid.resize(frames);

  #pragma omp declare reduction(framesReduction : vector<SizedImg> : framesReduction(omp_out, omp_in)) initializer (omp_priv=omp_orig)

  #pragma omp parallel for num_threads(threads) reduction(framesReduction : final_vid)
  for(int i = 0; i < frames; i++){
    SizedImg img(input_height, input_width, input_channels, (unsigned char*) vid_data[i].data);
    SizedImg new_img = resizeImg(img);

    final_vid[i] = new_img;
  }

  for(int i = 0; i < frames; i++){
    cv::Mat new_frame(output_height, output_width, CV_8UC3, final_vid[i].img);
    inout_videos.ss << new_frame;
    free(final_vid[i].img);
  }


  // Release both files
  inout_videos.ff.release();
  inout_videos.ss.release();

  cout << "Video processing completed." << endl;

  return 0;
}


void framesReduction(vector<SizedImg> &out, vector<SizedImg> &in){
  for(int i = 0; i < out.size() ;i++){
    if(out[i].height != 0) continue;
    if(in[i].height == 0) continue;
    SizedImg aux(in[i].height, in[i].width, in[i].channels, in[i].img);
    out[i] = in[i];
  }
}

// FUNCTIONS
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

SizedImg resizeImg(SizedImg &img){
  SizedImg new_img(output_height, output_width, input_channels, (unsigned char*) malloc(output_height * output_width * input_channels * sizeof(unsigned char)));

  unsigned int avg = 0;

  for(int i = 0; i < output_height; i++){
    for(int j = 0; j < output_width; j++){
      for(int k = 0; k < input_channels; k++){
        avg = 0;
        for(int dx = 0; dx < scale_factor; dx++){
          for(int dy = 0; dy < scale_factor; dy++){
            int x = i*scale_factor + dx;
            int y = j*scale_factor + dy;
            unsigned int tmp = (unsigned int) img.img[x*input_width*input_channels + y*input_channels + k];
            avg = avg + tmp;
          }
        }
  
        avg = avg/(scale_factor*scale_factor);

        new_img.img[i*output_width*input_channels + j*input_channels + k] = (unsigned char) avg;
      }
    }
  }

  return new_img;
}
