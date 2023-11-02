EXE := video_reduction
CUDAEXE := video_reduction_cuda

HDIR := headers
TRASHDIR := trash

CUDA := nvcc
CUDALIBS := `pkg-config --cflags --libs opencv4`

CXX := g++
CXXFLAGS := -g -Wall -pedantic -std=c++20
CXXFASTFLAGS := -O3

CPPFLAGS := -I$(HDIR)
CPPFLAGS += -MMD -MP
# Set local variable if something has to be defined locally
CPPFLAGS += -DLOCAL

OPENCVFLAGS := -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc
OPENMPFLAGS := -fopenmp

ifeq (build,$(firstword $(MAKECMDGOALS)))
	SRC := $(word 2, $(MAKECMDGOALS))
  $(eval $(SRC):;@:)
endif

ifeq (buildcuda,$(firstword $(MAKECMDGOALS)))
	CUDASRC := $(word 2, $(MAKECMDGOALS))
  $(eval $(CUDASRC):;@:)
endif

ifeq (clean,$(firstword $(MAKECMDGOALS)))
	SRC := 
endif

#ifeq ($(src),)
	#$(error Please specify the source file to build with 'make src=<filename>')
#endif

CUDASOURCES := $(CUDASRC)

SOURCES := $(SRC)
OBJ := $(patsubst %.cpp,$(TRASHDIR)/%.o,$(SOURCES))
DEP := $(OBJ:.o=.d)

build: $(EXE)

buildcuda: $(CUDAEXE)

$(CUDAEXE): $(CUDASOURCES)
	$(CUDA) $^ -o $@ $(CUDALIBS)

$(EXE): $(OBJ)
	$(CXX) $^ -o $@ $(CXXFASTFLAGS) $(OPENCVFLAGS) $(OPENMPFLAGS)

$(OBJ): $(SOURCES) | $(TRASHDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(CXXFASTFLAGS) $(OPENCVFLAGS) $(OPENMPFLAGS) -o $@ -c $< 

$(TRASHDIR):
	mkdir $@

clean:
	$(RM) -r $(TRASHDIR) $(EXE)

-include $(DEP)
