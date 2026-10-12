# Native GPU fractal explorer (gpu.cpp + SDL2).
#
#   make deps    clone gpu.cpp (pinned) into third_party/
#   make dawn    build + stage Dawn for gpu.cpp (needs cmake, ninja, several GB of disk)
#   make         build ./fractal_explorer
#   make run
#   make bench   headless benchmark / reference renderer (tools/bench.cpp)
#   make test    headless checks of the incremental frame pipeline
#
# Also needs MPFR + GMP for the deep-zoom reference orbits (brew install mpfr).

GPUCPP     := third_party/gpu.cpp
GPUCPP_REV := 09bdeaa7a0c6715ed7624ee90eda53aa2d8c4cbc
DAWN       := $(GPUCPP)/third_party/dawn
MPFR       ?= $(shell brew --prefix 2>/dev/null || echo /usr/local)
HEADERS    := src/fractal_shader.hpp src/viewport.hpp src/bigfloat.hpp src/perturbation.hpp \
              src/pane_renderer.hpp

CXXFLAGS := -std=c++20 -O3 -Wall -Wextra \
            -isystem $(GPUCPP) -isystem $(DAWN)/include -isystem $(MPFR)/include \
            $(shell sdl2-config --cflags)
MPFRLIBS := -L$(MPFR)/lib -lmpfr -lgmp
LDLIBS   := -L$(DAWN)/lib -lwebgpu_dawn -Wl,-rpath,$(abspath $(DAWN)/lib) \
            $(shell sdl2-config --libs) $(MPFRLIBS)

fractal_explorer: src/main.cpp $(HEADERS)
	@test -f $(DAWN)/include/dawn/webgpu_cpp.h || \
	  { echo "Dawn is not staged: run 'make deps dawn' first"; exit 1; }
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

deps: $(GPUCPP)

$(GPUCPP):
	git clone https://github.com/AnswerDotAI/gpu.cpp $@
	git -C $@ checkout --detach -q $(GPUCPP_REV)

dawn: $(GPUCPP)
	$(GPUCPP)/tools/build_dawn.sh

run: fractal_explorer
	./fractal_explorer

bench: tools/bench.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(DAWN)/lib -lwebgpu_dawn \
	  -Wl,-rpath,$(abspath $(DAWN)/lib) -lz $(MPFRLIBS)

pipeline_test: tools/pipeline_test.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(DAWN)/lib -lwebgpu_dawn \
	  -Wl,-rpath,$(abspath $(DAWN)/lib) $(MPFRLIBS)

test: pipeline_test
	./pipeline_test

clean:
	rm -f fractal_explorer bench pipeline_test

.PHONY: deps dawn run test clean
