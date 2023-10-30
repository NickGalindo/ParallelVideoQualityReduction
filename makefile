EXE := video_reduction

HDIR := headers
TRASHDIR := trash

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

ifeq (clean,$(firstword $(MAKECMDGOALS)))
	SRC := 
endif

#ifeq ($(src),)
	#$(error Please specify the source file to build with 'make src=<filename>')
#endif

SOURCES := $(SRC)
OBJ := $(patsubst %.cpp,$(TRASHDIR)/%.o,$(SOURCES))
DEP := $(OBJ:.o=.d)

build: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $^ -o $@ $(CXXFASTFLAGS) $(OPENCVFLAGS) $(OPENMPFLAGS)

$(OBJ): $(SOURCES) | $(TRASHDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(CXXFASTFLAGS) $(OPENCVFLAGS) $(OPENMPFLAGS) -o $@ -c $< 

$(TRASHDIR):
	mkdir $@

clean:
	$(RM) -r $(TRASHDIR) $(EXE)

-include $(DEP)
