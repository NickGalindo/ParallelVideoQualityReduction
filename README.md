# Practica 1 - Computacion Paralela - Reduccion de resolucion de un video 1080p a 360p
Entrega de la practica 1 de computacion parallela. El objetivo de esta entrega es utilizar OpenMp para paralelizar la reduccion de la resolucion de un video de 1080p a 360p.

## Dependencias
```
stdc++20
opencv4
OpenMp
```

## Files
`/headers` contains bits/stdc++.h used to import multiple useful libraries in one header

`/tmp` contains some test videos for video reduction

`/trash` saves the .o and .d files resulting from compilation

`README.md` classic readme

`compile_commands.json` file that tells clangd lsp how to interpret the codebase, not necessary for compilation or running

`makefile` makefile tells g++ how to compile and link

`results.txt` contains the results of running ./script_ejecutar_todo.sh

`script_ejecutar_todo.sh` usage: `./script_ejecutar_todo.sh input_video.mp4 output_path.mp4` This runs the program over 32, 16, 8, 4, 2, 1 threads on the input and output videos.

`video_reduction` the executable for the program. Usage: `./video_reduction input_video.mp4 output_path.mp4 #_threads`

`video_reduction.cpp` the code for video reduction. This code processes frames in parallel and syncornizes inside an omp critical pragma

`video_reduction_parallel_frame_resize.cpp` alternate code for reduction. This code processes frames in parallel and syncronizes using omp reduction.

`video_reduction_parallel_resizing.cpp` alternate code for reduction. This code resizes each frame in parallel. I.E. multiple threads are spawned per frame to make resizing that particular frame faster. This one show little improvement with increased threads but that can be explained because of the overhead that spawning new threads causes.

## Usage
### Compile
To just compile:
```
make build video_reduction.cpp
```

To compile and add clangd lsp inference
```
bear -- make build video_reduction.cpp
```

### Running
```
./video_reduction input_video.mp4 output_path.mp4 number_threads
```

### TO MAKE CLANGD RECOGNIZE bits/stdc++.h RUN THE FOLLOWING
```bear -- make build a.cpp```
