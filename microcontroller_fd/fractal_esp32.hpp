#ifndef FRACTAL_ESP32_H
#define FRACTAL_ESP32_H

#include <stdint.h>

// Image size
#define M 16
#define MAX_SCALES 4
#define G 256  // gray levels

// RGB image: [channel][row][col]
typedef uint8_t ImageRGB[3][M][M];

// Function declarations

// RGB Differential Box Counting
void DBC_RGB(ImageRGB img,
             unsigned long Nr[MAX_SCALES],
             int box_sizes[MAX_SCALES],
             int *num_scales);

// Fractal dimension computation
float compute_fractal_dimension(unsigned long Nr[MAX_SCALES],
                                int box_sizes[MAX_SCALES],
                                int num_scales,
                                int img_size);

// Main wrapper
float get_fractal_dimension(ImageRGB img);

#endif