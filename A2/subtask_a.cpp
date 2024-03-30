#include <vector>
using namespace std;

vector<vector<float>> convolve(const vector<vector<float>>& input, const vector<vector<float>>& kernel, int pad = 0) {
    int iRows = input.size();
    int iCols = input[0].size();
    int kRows = kernel.size();
    int kCols = kernel[0].size();
    
    // Output size
    int oRows = iRows - kRows + 1 + 2*pad;
    int oCols = iCols - kCols + 1 + 2*pad;
    
    vector<vector<float>> output(oRows, vector<float>(oCols, 0));
    
    for (int i = 0; i < oRows; ++i) {
        for (int j = 0; j < oCols; ++j) {
            for (int m = 0; m < kRows; ++m) {
                for (int n = 0; n < kCols; ++n) {
                    int iIdx = i + m - pad;
                    int jIdx = j + n - pad;
                    if (iIdx >= 0 && iIdx < iRows && jIdx >= 0 && jIdx < iCols) {
                        output[i][j] += input[iIdx][jIdx] * kernel[m][n];
                    }
                }
            }
        }
    }
    return output;
}

vector<vector<float>> relu(const vector<vector<float>>& input) {
    int rows = input.size();
    int cols = input[0].size();
    vector<vector<float>> output(rows, vector<float>(cols));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[i][j] = max(0.0f, input[i][j]);
        }
    }
    return output;
}

vector<vector<float>> tanhActivation(const vector<vector<float>>& input) {
    int rows = input.size();
    int cols = input[0].size();
    vector<vector<float>> output(rows, vector<float>(cols));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[i][j] = tanh(input[i][j]);
        }
    }
    return output;
}

vector<vector<float>> maxPooling(const vector<vector<float>>& input, int windowSize) {
    int rows = input.size();
    int cols = input[0].size();
    int oRows = rows / windowSize;
    int oCols = cols / windowSize;
    vector<vector<float>> output(oRows, vector<float>(oCols, 0));
    
    for (int i = 0; i < oRows; ++i) {
        for (int j = 0; j < oCols; ++j) {
            float maxVal = numeric_limits<float>::lowest();
            for (int m = 0; m < windowSize; ++m) {
                for (int n =0; n < windowSize; ++n) {
                    int iIdx = i * windowSize + m;
                    int jIdx = j * windowSize + n;
                    if (iIdx < rows && jIdx < cols) {
                        maxVal = max(maxVal, input[iIdx][jIdx]);
                    }
                }
            }
            output[i][j] = maxVal;
        }
    }
    return output;
}

vector<vector<float>> averagePooling(const vector<vector<float>>& input, int windowSize) {
    int rows = input.size();
    int cols = input[0].size();
    int oRows = rows / windowSize;
    int oCols = cols / windowSize;
    vector<vector<float>> output(oRows, vector<float>(oCols, 0));
    
    for (int i = 0; i < oRows; ++i) {
        for (int j = 0; j < oCols; ++j) {
            float sum = 0.0;
            for (int m = 0; m < windowSize; ++m) {
                for (int n = 0; n < windowSize; ++n) {
                    int iIdx = i * windowSize + m;
                    int jIdx = j * windowSize + n;
                    if (iIdx < rows && jIdx < cols) {
                        sum += input[iIdx][jIdx];
                    }
                }
            }
            output[i][j] = sum / (windowSize * windowSize);
        }
    }
    return output;
}

vector<float> softmax(const vector<float>& input) {
    vector<float> output(input.size());
    float maxVal = *max_element(input.begin(), input.end());
    float sum = 0.0f;

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i] - maxVal); // Improve numerical stability
        sum += output[i];
    }
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] /= sum;
    }
    
    return output;
}

vector<float> sigmoid(const vector<float>& input) {
    vector<float> output(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
    
    return output;
}
