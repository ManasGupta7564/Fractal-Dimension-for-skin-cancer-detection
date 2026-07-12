#include <Arduino.h>
#include "fractal_esp32.h"

// ---- CONFIG ----
#define H 64   // image height
#define W 64   // image width
#define STEP 4

// Full image (RGB)
uint8_t full_img[3][H][W];

// Patch (16x16)
ImageRGB patch;

// Output FD map
#define OUT_ROWS ((H - M) / STEP + 1)
#define OUT_COLS ((W - M) / STEP + 1)

float fractal_map[OUT_ROWS][OUT_COLS];


// ---- Dummy image loader (replace later) ----
void load_dummy_image() {
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                full_img[c][i][j] = random(0, 256);
            }
        }
    }
}


// ---- Extract 16x16 patch ----
void extract_patch(int x, int y) {
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                patch[c][i][j] = full_img[c][y + i][x + j];
            }
        }
    }
}


// ---- Sliding window FD computation ----
void compute_fractal_map() {

    int row_idx = 0;

    for (int y = 0; y <= H - M; y += STEP) {

        int col_idx = 0;

        for (int x = 0; x <= W - M; x += STEP) {

            extract_patch(x, y);

            float fd = get_fractal_dimension(patch);

            fractal_map[row_idx][col_idx] = fd;

            col_idx++;
        }

        row_idx++;
    }
}


// ---- Print map (instead of CSV) ----
void print_fractal_map() {
    for (int i = 0; i < OUT_ROWS; i++) {
        for (int j = 0; j < OUT_COLS; j++) {
            Serial.print(fractal_map[i][j], 4); // 4 decimal precision
            if (j < OUT_COLS - 1) Serial.print(",");
        }
        Serial.println();
    }
}


void setup() {
    Serial.begin(115200);

    // Step 1: Load image
    load_dummy_image();

    // Step 2: Compute FD map
    compute_fractal_map();

    // Step 3: Output (CSV-like)
    Serial.println("Fractal Map:");
    print_fractal_map();
}

void loop() {
    // nothing
}