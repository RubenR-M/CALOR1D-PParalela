#include <iostream>
#include <math.h>
using namespace std;

/**
 *  Compilacion:
 *   -> nvcc pileibniz_cuda.cu -o pileibniz_cuda
 */

__global__ void calcularPi(float *device_values)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    device_values[idx] = pow(-1, idx) / (2 * idx + 1);
}

int main(int argc, char *argv[])
{
    float sum = 0;
    float pi, *device_values, *host_values;
    unsigned long int iteraciones = 32 * atoi(argv[1]);

    cudaMalloc((void **)&device_values, iteraciones * sizeof(float));
    host_values = (float *)malloc(iteraciones * sizeof(float));

    calcularPi<<<iteraciones / 32, 32>>>(device_values);

    cudaMemcpy(host_values, device_values, iteraciones * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < iteraciones; i++)
    {
        sum += host_values[i];
    }
    pi = 4 * sum;
    cout << pi << endl;
    cudaFree(device_values);
    cudaFreeHost(host_values);
    return 0;
}