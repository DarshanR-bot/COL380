#include <vector>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
// Assume the existence of utility functions and CUDA kernel prototypes:
// loadWeights, convLayerKernel, reluKernel, maxPoolingKernel, fullyConnectedKernel
__global__ void conv3DKernelWithBias(float* input, float* output, float* kernel, float* biases,
                                                        int inputHeight, int inputWidth, int inputChannels,
                                                        int outputHeight, int outputWidth, int outputChannels,
                                                        int kernelHeight, int kernelWidth) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outZ = blockIdx.z * blockDim.z + threadIdx.z;

    if (outX < outputWidth && outY < outputHeight && outZ < outputChannels) {
        float value = 0.0f;

        for (int k = 0; k < inputChannels; ++k) {
            for (int offsetY = 0; offsetY < kernelHeight; ++offsetY) {
                for (int offsetX = 0; offsetX < kernelWidth; ++offsetX) {
                    int inX = outX + offsetX ; // Not Assuming kernel is centered over the pixel
                    int inY = outY + offsetY ;

                    if (inX >= 0 && inX < inputWidth && inY >= 0 && inY < inputHeight) {
                        // Adjust index for channel-consecutive storage
                        int inputIndex = (k * inputHeight * inputWidth) + (inY * inputWidth + inX);
                        
                        int kernelIndex = (outZ * inputChannels * kernelHeight * kernelWidth) + 
                                          (k * kernelHeight * kernelWidth) + 
                                          (offsetY * kernelWidth) + offsetX;
                        
                        value += input[inputIndex] * kernel[kernelIndex];
                    }
                }
            }
        }

        // Add bias and write to output, adjusting index for channel-consecutive storage
        int outputIndex = (outZ * outputHeight * outputWidth) + (outY * outputWidth + outX);
        output[outputIndex] = value + biases[outZ];
    }
}

__global__ void maxPoolingKernel(float* input, float* output, 
                                                    int inputHeight, int inputWidth, int channels,
                                                    int poolHeight, int poolWidth, 
                                                    int outputHeight, int outputWidth, int stride) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (outX < outputWidth && outY < outputHeight && channel < channels) {
        float maxVal = -FLT_MAX;  // Initialize to smallest float value
        for (int poolY = 0; poolY < poolHeight; ++poolY) {
            for (int poolX = 0; poolX < poolWidth; ++poolX) {
                int inX = outX * stride + poolX;
                int inY = outY * stride + poolY;
                if (inX < inputWidth && inY < inputHeight) {
                    // Calculate index for channel-consecutive input
                    int inputIndex = (channel * inputHeight * inputWidth) + (inY * inputWidth + inX);
                    maxVal = fmaxf(maxVal, input[inputIndex]);
                }
            }
        }

        // Write the max value to output, using channel-consecutive indexing
        int outputIndex = (channel * outputHeight * outputWidth) + (outY * outputWidth + outX);
        output[outputIndex] = maxVal;
    }
}

__global__ void reluKernel(float* inputOutput, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        inputOutput[idx] = fmaxf(0.0f, inputOutput[idx]);
    }
}



int main() {
    // Initialize dimensions based on the architecture
    // const int inputSize = 28 * 28;  // Input image size for MNIST
    // const int conv1OutputSize = 24 * 24 * 20; // Output size after first Conv layer
    // Similarly, calculate for other layers...

    // Allocate memory for input and output of each layer, weights, and biases
    // For simplicity, we're using single float pointers; in practice, consider using structures or classes for managing layers
    // float *d_input, *d_conv1Output, *d_conv1Weights, /* other layer outputs, weights, biases */;

    // Allocate GPU memory (cudaMalloc) for inputs, outputs, weights, biases

    // Load weights from files and transfer them to GPU memory (cudaMemcpy)

    // Setup kernel execution parameters (dim3 blockDim, dim3 gridDim) for each layer

    /// Starting by trying one convolution layer
    // Open the file
    std::ifstream file("./trained_weights/conv1.txt");
    std::ifstream img_file("./output.txt");
    
    if (!file.is_open() || !img_file.is_open()) {
        std::cerr << "Error: Unable to open file" << std::endl;
        return 1;
    }
    
    // Read the values into a vector
    std::vector<float> weights_and_bias;
    std::vector<float> img_dat;
    float value;
    
    while (img_file >> value) {
        img_dat.push_back(value);
    }
    while (file >> value) {
        weights_and_bias.push_back(value);
    }
    
    file.close();
    img_file.close();
    
    // Assuming input image dimensions and Conv_1 output dimensions
    const int inputWidth = 28;
    const int inputHeight = 28;
    const int inputChannels = 1; // Grayscale image
    const int outputWidth = 24;
    const int outputHeight = 24;
    const int outputChannels = 20; // Number of filters
    const int kernelHeight = 5;
    const int kernelWidth = 5;

    // Flatten input and output dimensions for easier memory allocation
    const int inputSize = inputWidth * inputHeight * inputChannels;
    const int outputSize = outputWidth * outputHeight * outputChannels;

    float *d_input, *d_output, *d_weights, *d_biases;

    // Allocate device memory
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));
    // For weights: 20 filters each of size 5x5, and 20 bias values
    cudaMalloc(&d_weights, outputChannels * kernelWidth * kernelHeight * sizeof(float));
    cudaMalloc(&d_biases, outputChannels * sizeof(float));


    cudaMemcpy(d_input, img_dat.data() , inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_and_bias.data(), outputChannels * kernelWidth * kernelHeight * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, weights_and_bias.data() + outputChannels * kernelWidth * kernelHeight, outputChannels * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockDim(16, 16, 1); // Keep z-dimension as 1 for simplicity in 2D convolutions

    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x,
                (outputHeight + blockDim.y - 1) / blockDim.y,
                outputChannels); // Ensure each output channel is handled
    // Launch the convolution kernel
    conv3DKernelWithBias<<<gridDim, blockDim>>>(d_input, d_output, d_weights, d_biases,
                                      inputHeight,  inputWidth,  inputChannels,
                                      outputHeight,  outputWidth,  outputChannels,
                                      kernelHeight,  kernelWidth);
    // Allocate memory for d_output on host
    std::vector<float> h_output(outputWidth * outputHeight * outputChannels);

    // Copy data from device to host
    cudaMemcpy(h_output.data(), d_output, outputWidth * outputHeight * outputChannels * sizeof(float), cudaMemcpyDeviceToHost);

    // Output file path
    const std::string output_file_path = "d_output.txt";

    // Open the output file
    std::ofstream output_file(output_file_path);

    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open output file" << std::endl;
        return 1;
    }

    // Print the d_output to the file
    for (int c = 0; c < outputChannels; ++c) {
        output_file << "Channel " << c << ":" << std::endl;
        for (int i = 0; i < outputHeight; ++i) {
            for (int j = 0; j < outputWidth; ++j) {
                output_file << h_output[(c * outputHeight * outputWidth) + (i * outputWidth) + j] << " ";
            }
            output_file << std::endl;
        }
        output_file << std::endl;
    }

    // Close the output file
    output_file.close();

    std::cout << "d_output written to file: " << output_file_path << std::endl;


    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
    ///

    // *** Execute Network Layers ***

    // Convolution Layer 1
    // convLayerKernel<<<gridDimConv1, blockDimConv1>>>(d_input, d_conv1Output, d_conv1Weights, /* other params */);
    // Apply ReLU Activation
    // reluKernel<<<gridDimRelu1, blockDimRelu1>>>(d_conv1Output, d_conv1Output, conv1OutputSize);

    // Pooling Layer 1 (Max Pooling)
    // maxPoolingKernel<<<gridDimPool1, blockDimPool1>>>(d_conv1Output, d_pool1Output, /* other params */);

    // Convolution Layer 2
    // Repeat the process for subsequent layers, matching the architecture specifics...

    // Fully Connected Layers - Consider handling these differently as they may not be direct convolutions
    // fullyConnectedKernel<<<gridDimFC1, blockDimFC1>>>(d_pool2Output, d_fc1Output, d_fc1Weights, /* other params */);
    // Apply ReLU for FC1

    // FC2 - Output Layer
    // Similar to FC1, adjust for the final output size and activation (if any)

    // Copy final layer output back to host
    // std::vector<float> h_output( /* size of the final output layer */ );
    // cudaMemcpy from d_fc2Output to h_output.data()

    // Apply Softmax on CPU for classification
    // std::vector<float> probabilities = softmax(h_output);  // Assuming softmax is defined on the host

    // Output the top probabilities or classification result

    // Cleanup: Free all allocated GPU memory

    return 0;
}
