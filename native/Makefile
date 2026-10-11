# Native GPU fractal explorer (gpu.cpp + SDL2).
#
#   make deps    clone gpu.cpp (pinned) into third_party/
#   make dawn    build + stage Dawn for gpu.cpp (needs cmake, ninja, several GB of disk)
#   make         build ./fractal_explorer
#   make run
#   make bench   headless benchmark / reference renderer (tools/bench.cpp)

GPUCPP     := third_party/gpu.cpp
GPUCPP_REV := 09bdeaa7a0c6715ed7624ee90eda53aa2d8c4cbc
DAWN       := $(GPUCPP)/third_party/dawn

CXXFLAGS := -std=c++20 -O3 -Wall -Wextra \
            -isystem $(GPUCPP) -isystem $(DAWN)/include $(shell sdl2-config --cflags)
LDLIBS   := -L$(DAWN)/lib -lwebgpu_dawn -Wl,-rpath,$(abspath $(DAWN)/lib) \
            $(shell sdl2-config --libs)

fractal_explorer: src/main.cpp src/fractal_shader.hpp src/viewport.hpp
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

bench: tools/bench.cpp src/fractal_shader.hpp src/viewport.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(DAWN)/lib -lwebgpu_dawn \
	  -Wl,-rpath,$(abspath $(DAWN)/lib) -lz

clean:
	rm -f fractal_explorer bench

.PHONY: deps dawn run clean
