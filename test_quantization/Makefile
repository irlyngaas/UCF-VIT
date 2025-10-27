# Makefile for rocBLAS INT8 test
# AMD MI250X / ROCm 6.2.4

# Use which to find hipcc
CXX = $(shell which hipcc || echo /opt/rocm-6.2.4/bin/hipcc)
CXXFLAGS = -std=c++17 -O3 -DNDEBUG
INCLUDES = -I$(ROCM_PATH)/include
LIBS = -L$(ROCM_PATH)/lib -lrocblas

# Default ROCm path - try different locations
ROCM_PATH ?= $(shell if [ -d "/opt/rocm-6.2.4" ]; then echo "/opt/rocm-6.2.4"; else echo "/opt/rocm"; fi)

TARGET_INT8 = rocblas_int8_test
TARGET_INT4 = rocblas_int4_test
TARGET_INT4_GPU = rocblas_int4_gpu_unpack
TARGET_INT4_ULTRA = rocblas_int4_ultra_optimized
TARGET_EXTREME = rocblas_extreme_quantization
TARGET_SYSTEMATIC = systematic_vectorization_analysis
TARGET_CORRECTED = corrected_vectorization_analysis
SOURCE_INT8 = rocblas_int8_test.cpp
SOURCE_INT4 = rocblas_int4_test.cpp
SOURCE_INT4_GPU = rocblas_int4_gpu_unpack.cpp
SOURCE_INT4_ULTRA = rocblas_int4_ultra_optimized.cpp
SOURCE_EXTREME = rocblas_extreme_quantization.cpp
SOURCE_SYSTEMATIC = systematic_vectorization_analysis.cpp
SOURCE_CORRECTED = corrected_vectorization_analysis.cpp

.PHONY: all clean test int8 int4 int4-gpu int4-ultra extreme systematic corrected

all: $(TARGET_INT8) $(TARGET_INT4) $(TARGET_INT4_GPU) $(TARGET_INT4_ULTRA) $(TARGET_EXTREME) $(TARGET_SYSTEMATIC) $(TARGET_CORRECTED)

int8: $(TARGET_INT8)

int4: $(TARGET_INT4)

int4-gpu: $(TARGET_INT4_GPU)

int4-ultra: $(TARGET_INT4_ULTRA)

extreme: $(TARGET_EXTREME)

systematic: $(TARGET_SYSTEMATIC)

corrected: $(TARGET_CORRECTED)

$(TARGET_INT8): $(SOURCE_INT8)
	@echo "Building rocBLAS INT8 test..."
	@echo "ROCm path: $(ROCM_PATH)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $(TARGET_INT8)"

$(TARGET_INT4): $(SOURCE_INT4)
	@echo "Building rocBLAS INT4 test..."
	@echo "ROCm path: $(ROCM_PATH)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $(TARGET_INT4)"

$(TARGET_INT4_GPU): $(SOURCE_INT4_GPU)
	@echo "Building rocBLAS INT4 GPU unpack test..."
	@echo "ROCm path: $(ROCM_PATH)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $(TARGET_INT4_GPU)"

$(TARGET_INT4_ULTRA): $(SOURCE_INT4_ULTRA)
	@echo "Building rocBLAS INT4 ULTRA optimized test..."
	@echo "ROCm path: $(ROCM_PATH)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $(TARGET_INT4_ULTRA)"

$(TARGET_EXTREME): $(SOURCE_EXTREME)
	@echo "Building rocBLAS EXTREME quantization suite..."
	@echo "ROCm path: $(ROCM_PATH)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $(TARGET_EXTREME)"

$(TARGET_SYSTEMATIC): $(SOURCE_SYSTEMATIC)
	@echo "Building SYSTEMATIC vectorization analysis..."
	@echo "ROCm path: $(ROCM_PATH)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $(TARGET_SYSTEMATIC)"

$(TARGET_CORRECTED): $(SOURCE_CORRECTED)
	@echo "Building CORRECTED vectorization analysis..."
	@echo "ROCm path: $(ROCM_PATH)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)
	@echo "Build complete: $(TARGET_CORRECTED)"

clean:
	rm -f $(TARGET_INT8) $(TARGET_INT4) $(TARGET_INT4_GPU) $(TARGET_INT4_ULTRA) $(TARGET_EXTREME) $(TARGET_SYSTEMATIC) $(TARGET_CORRECTED)

test: $(TARGET_INT8)
	@echo "Running rocBLAS INT8 test..."
	./$(TARGET_INT8)

test-int4: $(TARGET_INT4)
	@echo "Running rocBLAS INT4 test..."
	./$(TARGET_INT4)

test-int4-gpu: $(TARGET_INT4_GPU)
	@echo "Running rocBLAS INT4 GPU unpack test..."
	./$(TARGET_INT4_GPU)

test-int4-ultra: $(TARGET_INT4_ULTRA)
	@echo "Running rocBLAS INT4 ULTRA optimized test..."
	./$(TARGET_INT4_ULTRA)

test-extreme: $(TARGET_EXTREME)
	@echo "Running rocBLAS EXTREME quantization suite..."
	./$(TARGET_EXTREME)

test-systematic: $(TARGET_SYSTEMATIC)
	@echo "Running SYSTEMATIC vectorization analysis..."
	./$(TARGET_SYSTEMATIC)

test-corrected: $(TARGET_CORRECTED)
	@echo "Running CORRECTED vectorization analysis..."
	./$(TARGET_CORRECTED)

info:
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "Includes: $(INCLUDES)"
	@echo "Libraries: $(LIBS)"
	@echo "ROCm path: $(ROCM_PATH)"