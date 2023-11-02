#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <typeinfo>
#include <cuda_runtime.h>

#define ff first
#define ss second

#define GPU_GLOBAL_MEMORY_BYTES 6226378752

using namespace std;
typedef long long ll;

// ------------------------------------------------------
ll output_width, output_height, input_width, input_height, input_channels, threads, frames;
double fps, scale_factor;


// ------------------------------------------------------
pair<cv::VideoCapture, cv::VideoWriter>  validateInput(int argc, char* argv[]){
  // Check if number of arguments matches
  if(argc != 3){
    cerr << "INPUT ERROR: Usage -> video_reduction input_video_path.mpg output_video_path.mpg" << endl;
    exit(1);
  }

  // Get paths of input videos and output videos
  string input_video_path = argv[1];
  string output_video_path = argv[2];

  // Recieve input video
  cv::VideoCapture inputVideo(input_video_path);
  if(!inputVideo.isOpened()){
    cerr << "ERROR: Couldn't open input video" << endl;
    exit(1);
  }

  // Calculate video properties
  input_width = (ll) inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
  input_height = (ll) inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);

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

  // set input frames
  frames = ll(inputVideo.get(cv::CAP_PROP_FRAME_COUNT));
  input_channels = 3;

  return {inputVideo, outputVideo};
}

ll calculateMaxFrameMultiprocess(){
  ll a = ((sizeof(char)*input_channels*input_width*input_height) + (sizeof(char)*input_channels*output_width*output_height));
  ll b = GPU_GLOBAL_MEMORY_BYTES;
  return (b/a)/2;
}

// ------------------------------------------------------
__global__ void resizeBatchFrames(unsigned char* in_mat, unsigned char* out_mat, ll in_width, ll in_height, ll out_width, ll out_height, ll in_channels, ll sc_factor){
  ll idx = threadIdx.x + blockIdx.x * blockDim.x;
  ll cur_out_ptr = idx*out_width*in_channels;
  ll cur_in_ptr = idx*in_width*in_channels*sc_factor;

  for(ll i = 0, j = 0; i < out_width && j < in_width; i++, j += sc_factor){
    for(ll k = 0; k < in_channels; k++){
      unsigned int avg = 0;
      for(ll dy = 0; dy < sc_factor; dy++){
        for(ll dx = 0; dx < sc_factor; dx++){
          avg += (unsigned int) in_mat[cur_in_ptr+(dy*in_width*in_channels) + (j+dx)*in_channels + k];
        }
      }
      avg = avg/(sc_factor*sc_factor);
      out_mat[cur_out_ptr + i*in_channels + k] = (unsigned char) avg;
    }
  }
}


// ------------------------------------------------------
int main(int argc, char* argv[]){
  // Fast io
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  pair<cv::VideoCapture, cv::VideoWriter> inout_videos = validateInput(argc, argv);

  // calculate necessary preprocessing numbers
  ll num_batch_process_frames = calculateMaxFrameMultiprocess();
  unsigned char *inp = (unsigned char *) malloc(sizeof(char)*input_width*input_height*input_channels*num_batch_process_frames);
  unsigned char *out = (unsigned char *) malloc(sizeof(char)*output_width*output_height*input_channels*num_batch_process_frames);

  const ll total_threads = num_batch_process_frames * output_height;
  const ll blocks = total_threads/1024;
  const ll threads_per_block = 1024;
  //const ll blocks = 1;
  //const ll threads_per_block = 360;

  cudaError_t err = cudaSuccess;


  // Extract all frames from video
  for(ll i = 0; i < frames; i++){
    cv::Mat frame;
    inout_videos.ff >> frame;

    if(!frame.empty()){
      input_channels = frame.channels();

      for(ll x = 0; x < input_height; x++){
        for(ll y = 0; y < input_width; y++){
          for(ll k = 0; k < input_channels; k++){
            inp[(i%num_batch_process_frames)*input_height*input_width*input_channels + x*input_width*input_channels + y*input_channels + k] = frame.data[x*input_width*input_channels + y*input_channels + k];
          }
        }
      }
    }

    if((i+1) % num_batch_process_frames == 0 || (i+1) == frames || frame.empty()){
      ll cur_batch_size = min(frames, num_batch_process_frames);

      cout << "Processing batch: ";
      if(cur_batch_size < num_batch_process_frames) cout << ((i+1)/num_batch_process_frames)+1 << endl;
      else cout << (i+1)/num_batch_process_frames << endl;

      unsigned char *d_inp, *d_out;
      
      err = cudaMalloc((void **)&d_inp, sizeof(char)*input_width*input_height*input_channels*num_batch_process_frames);
      if(err != cudaSuccess){
        cerr << "Failed to allocate device d_inp: " << cudaGetErrorString(err);
        return 1;
      }

      err = cudaMalloc((void **)&d_out, sizeof(char)*output_width*output_height*input_channels*num_batch_process_frames);
      if(err != cudaSuccess){
        cerr << "Failed to allocate device d_out: " << cudaGetErrorString(err);
        return 1;
      }

      err = cudaMemcpy(d_inp, inp, sizeof(char)*input_width*input_height*input_channels*num_batch_process_frames, cudaMemcpyHostToDevice);
      if(err != cudaSuccess){
        cerr << "Failed to copy inp to d_inp: " << cudaGetErrorString(err);
        return 1;
      }

      err = cudaMemcpy(d_out, out, sizeof(char)*output_width*output_height*input_channels*num_batch_process_frames, cudaMemcpyHostToDevice);
      if(err != cudaSuccess){
        cerr << "Failed to copy out to d_out: " << cudaGetErrorString(err);
        return 1;
      }

      cout << "Processing Batch in GPU" << endl;
      resizeBatchFrames<<<blocks, threads_per_block>>>(d_inp, d_out, (const ll) input_width, (const ll) input_height, (const ll) output_width, (const ll) output_height, (const ll) input_channels, (const ll) scale_factor);
      cout << "Finished Processing Batch in GPU" << endl;

      err = cudaGetLastError();
      if(err != cudaSuccess){
        cerr << "Failed to run the gpu multiprocess: " << cudaGetErrorString(err);
        return 1;
      }

      err = cudaMemcpy(out, d_out, sizeof(char)*output_height*output_width*input_channels*num_batch_process_frames,cudaMemcpyDeviceToHost);
      if(err != cudaSuccess){
        cerr << "Failed to copy from device to host d_out to out: " << cudaGetErrorString(err);
        return 1;
      }


      for(int j = 0; j < cur_batch_size; j++){
        unsigned char *aux_img = (unsigned char *) malloc(sizeof(char)*output_height*output_width*input_channels);

        memcpy(aux_img, out + j*output_width*output_height*input_channels, sizeof(char)*output_width*output_height*input_channels);

        cv::Mat new_frame(output_height, output_width, CV_8UC3, aux_img);
        inout_videos.ss << new_frame;
        free(aux_img);
      }

      err = cudaFree(d_inp);
      if(err != cudaSuccess){
        cerr << "Failed to free device memory d_inp: " << cudaGetErrorString(err);
        return 1;
      }

      err = cudaFree(d_out);
      if(err != cudaSuccess){
        cerr << "Failed to free device memory d_out: " << cudaGetErrorString(err);
        return 1;
      }

      cout << "Finished Processing batch: ";
      if(cur_batch_size < num_batch_process_frames) cout << ((i+1)/num_batch_process_frames)+1 << endl;
      else cout << (i+1)/num_batch_process_frames << endl;
    }

    if(frame.empty()) break;
  }

  free(inp);
  free(out);

  err = cudaDeviceReset();
  if(err != cudaSuccess){
    cerr << "Failed to deinitialize device: " << cudaGetErrorString(err);
    return 1;
  }

  // Release both files
  inout_videos.ff.release();
  inout_videos.ss.release();

  cout << "Video processing completed." << endl;

  return 0;
}


