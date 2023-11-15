/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include <opencv2/opencv.hpp>
#include <vector>

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

void drawLine(cv::Mat &img, double rho, double theta, cv::Scalar color) {
    cv::Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    line(img, pt1, pt2, color, 3, cv::LINE_AA);
}


__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <image.pgm>\n", argv[0]);
        return -1;
    }

    PGMImage inImg(argv[1]);
    int w = inImg.x_dim;
    int h = inImg.y_dim;
    int *cpuht;

    // Reserva memoria en el host y el device
    float* d_Cos;
    float* d_Sin;
    unsigned char *d_in;
    int *d_hough;
    cudaMalloc((void**)&d_Cos, sizeof(float) * degreeBins);
    cudaMalloc((void**)&d_Sin, sizeof(float) * degreeBins);
    cudaMalloc((void**)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void**)&d_hough, sizeof(int) * degreeBins * rBins);
    int* h_hough = (int*)malloc(degreeBins * rBins * sizeof(int));

    // Calcula la transformada de Hough en la CPU
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    // Calcula rMax y rScale basado en el tamaño de la imagen
    float rMax = sqrtf(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // Configura el número de bloques e hilos
    int threadsPerBlock = 256;
    int blockNum = (w * h + threadsPerBlock - 1) / threadsPerBlock;

    // Define eventos para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inicia la medición del tiempo
    cudaEventRecord(start);

    // Llama al kernel
    GPU_HoughTran<<<blockNum, threadsPerBlock>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

    // Finaliza la medición del tiempo
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcula el tiempo transcurrido
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Hough Transform took %f milliseconds\n", milliseconds);

    // Copia los resultados de vuelta al host
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Carga la imagen original y conviértela a color
    cv::Mat originalImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat colorImage;
    cvtColor(originalImage, colorImage, cv::COLOR_GRAY2BGR);

    // Encuentra líneas que superan el umbral y dibújalas
    float sum = 0;
    float sumSq = 0;
    int count = 0;
    for (int i = 0; i < degreeBins * rBins; i++) {
        sum += h_hough[i];
        sumSq += h_hough[i] * h_hough[i];
        count++;
    }
    float promedio = sum / count;
    float varianza = (sumSq / count) - (promedio * promedio);
    float desviacionEstandar = sqrt(varianza);
    float threshold = promedio + 2 * desviacionEstandar;

    for (int r = 0; r < rBins; r++) {
        for (int t = 0; t < degreeBins; t++) {
            if (h_hough[r * degreeBins + t] > threshold) {
                float rho = (r - rBins / 2) * rScale;
                float theta = t * radInc;
                drawLine(colorImage, rho, theta, cv::Scalar(0, 0, 255));
            }
        }
    }

    // Guarda la imagen resultante
    cv::imwrite("hough_lines.jpg", colorImage);

    // Limpieza
    cudaFree(d_Cos);
    cudaFree(d_Sin);
    cudaFree(d_in);
    cudaFree(d_hough);
    free(h_hough);
    delete[] cpuht;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
