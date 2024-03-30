__global__ void convolveKernel(float *input, float *output, float *kernel, int iRows, int iCols, int kRows, int kCols, int pad) {
    int oRow = blockIdx.y * blockDim.y + threadIdx.y;
    int oCol = blockIdx.x * blockDim.x + threadIdx.x;
    int oRows = iRows - kRows + 1 + 2*pad;
    int oCols = iCols - kCols + 1 + 2*pad;

    if (oRow < oRows && oCol < oCols) {
        float value = 0.0f;
        for (int m = 0; m < kRows; ++m) {
            for (int n = 0; n < kCols; ++n) {
                int iIdx = oRow + m - pad;
                int jIdx = oCol + n - pad;
                if (iIdx >= 0 && iIdx < iRows && jIdx >= 0 && jIdx < iCols) {
                    value += input[iIdx * iCols + jIdx] * kernel[m * kCols + n];
                }
            }
        }
        output[oRow * oCols + oCol] = value;
    }
}

__global__ void reluKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(0.0f, input[idx]);
    }
}

__global__ void tanhKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]); // Use the tanhf function for single precision
    }
}


__global__ void maxPoolingKernel(float *input, float *output, int iRows, int iCols, int windowSize, int oRows, int oCols) {
    int oRow = blockIdx.y * blockDim.y + threadIdx.y;
    int oCol = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (oRow < oRows && oCol < oCols) {
        float maxVal = -FLT_MAX;
        for (int m = 0; m < windowSize; ++m) {
            for (int n = 0; n < windowSize; ++n) {
                int iIdx = oRow * windowSize + m;
                int jIdx = oCol * windowSize + n;
                if (iIdx < iRows && jIdx < iCols) {
                    maxVal = max(maxVal, input[iIdx * iCols + jIdx]);
                }
            }
        }
        output[oRow * oCols + oCol] = maxVal;
    }
}

__global__ void averagePoolingKernel(float *input, float *output, int iRows, int iCols, int windowSize, int oRows, int oCols) {
    int oRow = blockIdx.y * blockDim.y + threadIdx.y;
    int oCol = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (oRow < oRows && oCol < oCols) {
        float sum = 0.0f;
        for (int m = 0; m < windowSize; ++m) {
            for (int n = 0; n < windowSize; ++n) {
                int iIdx = oRow * windowSize + m;
                int jIdx = oCol * windowSize + n;
                if (iIdx < iRows && jIdx < iCols) {
                    sum += input[iIdx * iCols + jIdx];
                }
            }
        }
        output[oRow * oCols + oCol] = sum / (windowSize * windowSize);
    }
}

__global__ void expKernel(float *input, float *output, float maxVal, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx] - maxVal); // Subtract maxVal to improve numerical stability
    }
}

// This kernel assumes that the sum of the exponentiated values has been computed on the host and copied to the device
__global__ void normalizeKernel(float *input, float *output, float sum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / sum;
    }
}

__global__ void sigmoidKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}


