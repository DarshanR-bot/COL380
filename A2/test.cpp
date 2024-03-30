#include <vector>
#include <iostream>
#include <fstream>
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



int main() {
    // Open the image file
    std::ifstream image_file("output.txt");
    
    if (!image_file.is_open()) {
        std::cerr << "Error: Unable to open image file" << std::endl;
        return 1;
    }
    
    // Read the image data into a vector of vectors
    const int image_rows = 28;
    const int image_cols = 28;
    std::vector<std::vector<float>> image_data(image_rows, std::vector<float>(image_cols));
    
    for (int i = 0; i < image_rows; ++i) {
        for (int j = 0; j < image_cols; ++j) {
            if (!(image_file >> image_data[i][j])) {
                std::cerr << "Error: Unable to read pixel value" << std::endl;
                return 1;
            }
        }
    }
    
    image_file.close();
    
    // Open the kernel file
    std::ifstream kernel_file("trained_weights/conv1.txt");
    
    if (!kernel_file.is_open()) {
        std::cerr << "Error: Unable to open kernel file" << std::endl;
        return 1;
    }
    
    // Read the kernel data into a vector of vectors
    const int kernel_size = 5;
    std::vector<std::vector<float>> kernel(kernel_size, std::vector<float>(kernel_size));
    
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            if (!(kernel_file >> kernel[i][j])) {
                std::cerr << "Error: Unable to read kernel value" << std::endl;
                return 1;
            }
        }
    }
    
    kernel_file.close();
    vector<vector<float>> output = convolve(image_data, kernel, 0);
    // Print the read image data
    std::cout << "Output:" << std::endl;
    for (int i = 0; i < 24; ++i) {
        for (int j = 0; j < 24; ++j) {
            std::cout << output[i][j] - 0.270575 << " ";
        }
        std::cout << std::endl;
    }
    


}