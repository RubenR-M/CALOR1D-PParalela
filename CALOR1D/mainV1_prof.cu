#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include "gnuplot.h"
#include <fstream>
#include <sstream>
#include "nvToolsExt.h"
#include "math.h"
using namespace std;

int Nt = 93;
int Nx = 10;
float dt = 0.01;
float dx = 0.1;
float T_0[] = {20, 30, 0, 0, 0, 0, 0, 0, 0, 30, 20};
float T_izq = 50, T_der = 50;
float k = 0.16;
/*
 * nvcc mainV1_prof.cu -o mainV1_prof -arch=sm_50 -lnvToolsExt
 * nsys profile -o mainV1 mainV1_prof
 */
__global__ void temperatura(float *d_T, float *d_t0, int Nx, int Nt, float T_der, float T_izq, float k)
{
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    if (idx == Nx - 1 && idy > 0)
    {
        d_T[threadIdx.y * Nx + threadIdx.x] = T_izq;
    }
    if (idx == 0 && idy > 0)
    {
        d_T[threadIdx.y * Nx + threadIdx.x] = T_der;
    }
    if (idy == 0)
    {
        d_T[threadIdx.y * Nx + threadIdx.x] = d_t0[threadIdx.x];
    }
    __syncthreads();
    if (idy > 0 && (idx > 0 && idx < Nx - 1))
    {
        // T[f * Nx + c] = T[(f - 1) * Nx + c] + k * (T[(f - 1) * Nx + c + 1] - 2 * T[(f - 1) * Nx + c] + T[(f - 1) * Nx + c - 1]);

        for (int i = 1; i < Nt; i++)
        {

            float a = (float)(d_T[(i - 1) * Nx + threadIdx.x]);
            float b = (float)(d_T[(i - 1) * Nx + threadIdx.x + 1]);
            float c = (float)(d_T[(i - 1) * Nx + threadIdx.x]);
            float d = (float)(d_T[(i - 1) * Nx + threadIdx.x - 1]);

            d_T[i * Nx + threadIdx.x] = (float)(a + k * (b - 2 * c + d));
        }
    }
}

void read(string path)
{
    vector<double> T_0;
    string strT_0;

    path = path;
    ifstream fin;
    fin.open(path);
    if (fin.is_open())
    {
        fin >> Nt >> Nx >> dt >> dx >> strT_0 >> T_izq >> T_der >> k;
        fin.close();
    }

    while (1) // Use a while loop, "i" isn't doing anything for you
    {
        if (strT_0.find(',') != std::string::npos) // if comman not found find return string::npos
        {
            double value;
            istringstream(strT_0) >> value;
            T_0.push_back(value);
            strT_0.erase(0, strT_0.find(',') + 1); // Erase all element including comma
        }
        else
            break; // Come out of loop
    }
}

void graph(float *h_T, vector<float> X, int Nx, int Nt)
{
    string nombreArchivo = "datos.dat";
    ofstream archivo;

    cout << setprecision(3) << fixed;
    cout << "t/x      | ";
    for (int i = 0; i < Nx; i++) // creacion del vector de longitudes espaciado
    {
        if (i == 0 || i == Nx - 1)
        {
            cout << to_string(X[i]) << "000  ";
        }
        else
        {
            cout << to_string(X[i]) << "00  ";
        }
    }
    cout << endl;
    cout << "------------------------------------------------------------------" << endl;

    vector<float> tiempo;
    for (int i = 0; i < Nx * Nt; i++)
    {
        if (i == 0)
        {
            cout << to_string(i * dt) << " | ";
        }
        if (i == 0 || i == Nx - 1)
        {
            cout << to_string(h_T[i]) << "00  ";
        }
        else if (to_string(h_T[i]).size() < 12)
        {
            cout << to_string(h_T[i]) << "0  ";
        }
        else
        {
            cout << to_string(h_T[i]) << "  ";
        }

        if (i != Nx * Nt - 1 && (i + 1) % Nx == 0)
        {
            if (tiempo.size() == 0)
            {
                tiempo.push_back(0);
            }
            cout << endl;
            cout << to_string(((i + 1) * dt) / Nx) << " | ";
            tiempo.push_back(((i + 1) * dt) / Nx);
        }
    }
    cout << endl;

    for (int i = 0; i < (Nt)-1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            X.push_back(j * dx);
        }
    }
    vector<float> t;
    for (int j = 0; j < Nt; j++)
    {
        for (int i = 0; i < Nx; i++)
        {
            t.push_back(tiempo[j]);
        }
    }
    archivo.open(nombreArchivo.c_str(), fstream::out);
    for (int i = 0; i < X.size(); i++)
    {
        archivo << X[i] << " " << t[i] << " " << h_T[i] << endl;
    }
    archivo.close();

    gnuplot p;
    p("set view map");
    p("set dgrid3d");
    p("set pm3d interpolate 0,0");
    p("splot 'datos.dat' using 1:2:3 with pm3d");
}

int main(int argc, char **argv)
{
    cudaEvent_t start, stop;
    if (argc > 1)
    {
        string path = (string)argv[1];
        read(path);
    }
    else
    {
        cout << "No entry found!!\nDefault Values: " << endl;
        cout << "Nt: " << Nt << endl
             << "Nx: " << Nx << endl
             << "dt: " << dt << endl
             << "dx: " << dx << endl
             << "T_der: " << T_der << endl
             << "T_izq: " << T_izq << endl
             << "k: " << k << endl;
        cout << "Tempaturas iniciales" << endl;
        cout << "[ ";
        for (int i = 0; i < Nx + 1; i++)
        {
            printf("%f ", T_0[i]);
        }
        cout << "]" << endl;
    }
    Nx = Nx + 1;
    // vector de la vara lleno de 0's
    float *h_T;
    // vector de las temperaturas iniciales
    float *h_t0;
    float millis;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    nvtxRangePush("generate data cpu");
    cudaEventRecord(start);
    cudaMallocHost((void **)&h_T, sizeof(float) * Nx * Nt);
    cudaMallocHost((void **)&h_t0, sizeof(float) * Nx);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    nvtxRangePop();
    cudaEventElapsedTime(&millis, start, stop);

    vector<float> X;
    for (int j = 0; j < Nx; ++j)
    {
        X.push_back(j * dx); // PARA GRAFICAR
        h_t0[j] = T_0[j];
    }

    for (int i = 0; i < Nt; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            h_T[i * Nx + j] = 0.0;
        }
    }
    // vector de la vara lleno de 0's en Grafica
    float *d_T;
    // vector de las temperaturas en Grafica
    float *d_t0;

    nvtxRangePush("generate data gpu");
    cudaEventRecord(start);
    cudaMalloc((void **)&d_T, sizeof(float) * Nx * Nt);
    cudaMalloc((void **)&d_t0, sizeof(float) * Nx);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    nvtxRangePop();
    cudaEventElapsedTime(&millis, start, stop);

    // cudaEventRecord(start);
    nvtxRangePush("Transfer to GPU");
    cudaEventRecord(start);
    cudaMemcpy(d_T, h_T, sizeof(float) * Nx * Nt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t0, h_t0, sizeof(float) * Nx, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    nvtxRangePop();
    cudaEventElapsedTime(&millis, start, stop);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);

    // cudaEventElapsedTime(&millis, start, stop);
    // unsigned int grid_rows = Nt;
    // unsigned int grid_cols = Nx;

    dim3 dimGrid(Nx, Nt);
    nvtxRangePush("Calculating Temperature");
    cudaEventRecord(start);
    temperatura<<<1, dimGrid>>>(d_T, d_t0, Nx, Nt, T_der, T_izq, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    nvtxRangePop();
    cudaEventElapsedTime(&millis, start, stop);
    // cudaDeviceSynchronize(); //No necesario por el EventSync
    

    

    
    nvtxRangePush("GPU to CPU");
    cudaEventRecord(start);
    cudaMemcpy(h_T, d_T, sizeof(float) * Nx * Nt, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    nvtxRangePop();
    cudaEventElapsedTime(&millis, start, stop);

    graph(h_T, X, Nx, Nt);
    nvtxRangePush("free memory");
    cudaEventRecord(start);
    cudaFree(d_T);
    cudaFree(d_t0);
    cudaFreeHost(h_T);
    cudaFreeHost(h_t0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    nvtxRangePop();
    cudaEventElapsedTime(&millis, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
