#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define ROW 10
#define COL 10
#define SPARSITY 80
#define BLOCK_SIZE 16

__global__ void matrix_mul(int* a, int* b, int* c, int row, int col) {
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIdx < row && colIdx < col) {
        int tmp = 0;
        for (int i = 0; i < col; i++) {
            tmp += a[rowIdx * col + i] * b[i * col + colIdx];
        }
        c[rowIdx * col + colIdx] = tmp;
    }
}

void initilizeMatrix(int* m, int row, int col) {
    float dense_rate = (100.0 - float(SPARSITY)) / 100.0;
    int total_elements = row * col;
    int dense_elements = dense_rate * total_elements;

    //srand(time(0));
    int i = 0;
    while (i < dense_elements) {
        int rnd_row = rand() % row;
        int rnd_col = rand() % col;
        int index = rnd_row * col + rnd_col;

        if (m[index] == 0) {
            m[index] = (rand() % 100) + 1;
            i++;
        }
    }
}

void printMatrix(int* m, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%5d ", m[i * col + j]);
        }
        printf("\n");
    }
}

int main() {
    int* h_a, * h_b, * h_c;
    int* d_a, * d_b, * d_c;

    h_a = (int*)malloc(sizeof(int) * ROW * COL);
    h_b = (int*)malloc(sizeof(int) * COL * ROW);
    h_c = (int*)malloc(sizeof(int) * ROW * ROW);

    memset(h_a, 0, sizeof(int) * ROW * COL);
    memset(h_b, 0, sizeof(int) * COL * ROW);
    memset(h_c, 0, sizeof(int) * ROW * ROW);

    initilizeMatrix(h_a, ROW, COL);
    initilizeMatrix(h_b, COL, ROW);

    printf("Matrix A:\n");
    printMatrix(h_a, ROW, COL);
    printf("\nMatrix B:\n");
    printMatrix(h_b, COL, ROW);

    cudaMalloc((void**)&d_a, sizeof(int) * ROW * COL);
    cudaMalloc((void**)&d_b, sizeof(int) * COL * ROW);
    cudaMalloc((void**)&d_c, sizeof(int) * ROW * ROW);

    cudaMemcpy(d_a, h_a, sizeof(int) * ROW * COL, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * COL * ROW, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((ROW + dimBlock.x - 1) / dimBlock.x, (ROW + dimBlock.y - 1) / dimBlock.y);

    matrix_mul << <dimGrid, dimBlock >> > (d_a, d_b, d_c, ROW, ROW);

    cudaMemcpy(h_c, d_c, sizeof(int) * ROW * ROW, cudaMemcpyDeviceToHost);

    printf("\nMatrix C:\n");
    printMatrix(h_c, ROW, ROW);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
