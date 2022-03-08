#include <iostream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <string>
#include "gnuplot.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
using namespace std;
#include <unistd.h>
/**
 *
 * compile:
 *          nvcc exp.cu -o exp -Xcompiler -fopenmp
 *
 * run:
 *          ./exp filename.txt
 *
 *     example:
 *          ./exp entradas.txt
 *
 * content filename.txt:
 *                      maxParticles
 *                      maxTimeSteps
 *                      a
 *                      b
 *                      dt
 *
 */

/**
 * Este kernel realiza los calculos para cada particula en un rango de tiempo
 *
 * el calculo se realiza usando la siguiente ecuacion:
 *
 *      y_(i+1) = y_i + dt*y'(t) -> t = i*dt
 *
 *
 * @param device_values_x
 * @param device_values_y
 * @param dt
 * @param maxTimeSteps
 * @return __global__
 */
__global__ void integration(float *device_values_x, float *device_values_y, float dt, int maxTimeSteps)
{
    // unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //  y1 = feval(funcion, x, y);
    //  hy1 = h * y1;
    //  fprintf('\n%2.0f %10.6f %10.6f', i, x, y)
    //      y = y + hy1;
    //  x = x + h;

    for (int i = 1; i < maxTimeSteps + 1; i++)
    {
        float t = i * dt;
        device_values_x[i] = device_values_x[i - 1] + dt * pow(sin(t), 2);
        device_values_y[i] = device_values_y[i - 1] + dt * pow(cos(t), 2);
    }
}

/**
 * Valores por defecto si no hay entrada
 */
int maxParticles = 10;
int maxTimeSteps = 100;
float a = 0, b = 10;
float dt = 0.1;
float dx = 0.1;
float dy = 0.1;

typedef struct
{
    float x;
    float y;
    float t;
} posParticle;

void generateSeeds(posParticle *seeds, const unsigned int maxParticles)
{
    unsigned int num_seeds = maxParticles;
    for (int i = 0; i < num_seeds; ++i)
    {
        const float radius = 0.1;
        const float alpha = 2.0f * M_PI * (float)i / (float)num_seeds;
        seeds[i].t = 0.0;
        seeds[i].x = 0.5f + radius * cos(alpha);
        seeds[i].y = 0.5f + radius * sin(alpha);
    }
}

/**
 * @brief generador de archivo con los puntos encontrados para cada particua
 *
 * @param host_values_x
 * @param host_values_y
 * @param maxTimeSteps
 * @param num_seed
 */
void graph(float *host_values_x, float *host_values_y, int maxTimeSteps, string num_seed)
{
    string nombreArchivo = "datos" + num_seed + ".dat";
    ofstream archivo;
    archivo.open(nombreArchivo.c_str(), fstream::out);
    for (int i = 0; i < maxTimeSteps; i++)
    {
        archivo << host_values_x[i] << ' ' << host_values_y[i] << endl;
    }
    archivo.close();
}

/**
 * @brief El objetivo de esta funcion funcion concatenar archivo tras archivos
 *        lo logra hasta un cierto archivo y luego ocurre un error.
 * @param file_name1
 * @param file_name2
 * @param i
 */
void concatenate(string file_name1, string file_name2, int i)
{
    ifstream fin1, fin2;
    ofstream fout;

    vector<string> lines1;
    vector<string> lines2;
    string line1;
    string line2;
    string file_name3 = "data.dat";
    fin1.open(file_name1);
    fin2.open(file_name2);
    fout.open(file_name3);
    while (getline(fin1, line1))
    {
        lines1.push_back(line1);
    }
    while (getline(fin2, line2))
    {
        lines2.push_back(line2);
    }
    for (int i = 0; i < maxTimeSteps; i++)
    {
        fout << lines1[i] << ' ' << lines2[i] << endl;
    }
    fin1.close();
    fin2.close();
    fout.close();
    std::remove(file_name1.c_str());
    std::remove(file_name2.c_str());
    rename(file_name3.c_str(), file_name2.c_str());
}

/**
 * @brief En esta funcion se hacen los llamados a metodos, graficos y se asignan los valores correspondientes si hay entradas
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char **argv)
{
    // float sum = 0;
    float *device_values_x, *device_values_y;
    float *host_values_x, *host_values_y;
    // float firt_values_x, firt_values_y;
    // float firt_values_2x, firt_values_2y;
    // float last_values_x, last_values_y;
    // float last_values_2x, last_values_2y;
    // unsigned long int m = maxTimeSteps;
    /**
     * @brief si hay una entrada remplaze, si no muestre los valores por defecto
     */
    if (argc > 1)
    {
        string path = (string)argv[1];
        ifstream fin;
        fin.open(path);
        if (fin.is_open())
        {
            fin >> maxParticles >> maxTimeSteps >> a >> b >> dt;
            fin.close();
        }
    }
    else
    {
        cout << "No entry found!!\nDefault Values: " << endl;
        cout << "maxParticles: " << maxParticles << endl
             << "maxTimeSteps: " << maxTimeSteps << endl
             << "a: " << a << endl
             << "b: " << b << endl
             << "dt: " << dt << endl;
    }
    int h = (b - a) / maxTimeSteps;

    /**
     * @brief generacion de particulas
     *
     */
    posParticle *seeds = (posParticle *)malloc(maxParticles * sizeof(posParticle));
    generateSeeds(seeds, maxParticles);

    /**
     * @brief este archivo es para confirmar que los puntos iniciales hayan sido asignado correctamente
     *
     */
    // string seedsfile = "seeds.dat";
    // ofstream archivoseeds;
    // archivoseeds.open(seedsfile.c_str(), fstream::out);

    // cudaMallocHost((void **)&host_values_x, sizeof(float) * maxTimeSteps);
    // cudaMallocHost((void **)&host_values_y, sizeof(float) * maxTimeSteps);
    // for (int i = 0; i < maxParticles; i++)
    // {
    //     host_values_x[i] = seeds[i].x;
    //     host_values_y[i] = seeds[i].y;
    // }
    // graph(host_values_x, host_values_y, maxTimeSteps, to_string(90));
    // gnuplot p;

    // p("plot 'datos90.dat' using 1:2 title ''");

    /**
     * @brief en ente for se inicializa el kernel con los vectores para cada particula
     * en el que los values_eje son los valores calculados
     *
     */
    for (int i = 0; i < maxParticles; i++)
    {
        cudaMallocHost((void **)&host_values_x, sizeof(float) * maxTimeSteps);
        cudaMallocHost((void **)&host_values_y, sizeof(float) * maxTimeSteps);

        // cudaStream_t stream[1];
        // cudaStreamCreate(&stream[1]);

        cudaMalloc((void **)&device_values_x, sizeof(float) * maxTimeSteps);
        cudaMalloc((void **)&device_values_y, sizeof(float) * maxTimeSteps);

        host_values_x[0] = seeds[i].x;
        host_values_y[0] = seeds[i].y;

        cudaMemcpy(device_values_x, host_values_x, maxTimeSteps * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_values_y, host_values_y, maxTimeSteps * sizeof(float), cudaMemcpyHostToDevice);

        // cudaMemcpyAsync(device_values_x, host_values_x, maxTimeSteps * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
        integration<<<ceil(maxTimeSteps / 1024) + 1, 1024>>>(device_values_x, device_values_y, dt, maxTimeSteps);
        // , 0, stream[1]
        // cudaMemcpyAsync(host_values_x, device_values_x, maxTimeSteps * sizeof(float), cudaMemcpyDeviceToHost, stream[1]);

        cudaMemcpy(host_values_x, device_values_x, maxTimeSteps * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_values_y, device_values_y, maxTimeSteps * sizeof(float), cudaMemcpyDeviceToHost);
        // archivoseeds << seeds[i].x << ' ' << seeds[i].y << endl;
        // if (i == 0)
        // {
        //     firt_values_x = host_values_x[0];
        //     firt_values_y = host_values_y[0];
        //     last_values_x = host_values_x[maxTimeSteps - 1];
        //     last_values_y = host_values_y[maxTimeSteps - 1];
        // }
        // if (i == ceil((maxTimeSteps - 1) / 2))
        // {
        //     firt_values_2x = host_values_x[0];
        //     firt_values_2y = host_values_y[0];
        //     last_values_2x = host_values_x[maxTimeSteps - 1];
        //     last_values_2y = host_values_y[maxTimeSteps - 1];
        // }

        graph(host_values_x, host_values_y, maxTimeSteps, to_string(i));
        cudaFreeHost(host_values_x);
        cudaFreeHost(host_values_y);
        cudaFree(device_values_x);
        cudaFree(device_values_y);
    }
    /**
     * @brief intento de concatenar los valores para tener un solo archivo
     *
     */
    // cout << "pruebas\n";
    // gnuplot p;
    // string filename;
    // string filename2;
    // for (int i = 0; i < maxParticles - 1; i++)
    // {
    //     filename = "datos" + to_string(i) + ".dat";
    //     filename2 = "datos" + to_string(i + 1) + ".dat";
    //     concatenate(filename, filename2, i);
    // }
    // // string plotline = "plot '" + filename2 + "' using 1 : 2 title ''";
    // cout << "pruebas\n";
    // // p(plotline);

    // string plotline = "plot ";
    // for (int i = 1; i <= maxParticles; i = i + 2)
    // {
    //     plotline = plotline + "'" + filename2 + "'" + " using " + to_string(i) + ":" + to_string(i + 1) + " title '',\\\n";
    // }

    // p(plotline);
    // printf("%f %f %f %f \n", firt_values_x, firt_values_y, firt_values_2x, firt_values_2y);
    // printf("%f %f %f %f \n", last_values_x, last_values_y, last_values_2x, last_values_2y);
    // float h1 = (firt_values_2x + firt_values_x) / (2);
    // float k1 = (firt_values_2y + firt_values_y) / (2);
    // float h2 = (last_values_2x + last_values_x) / (2);
    // float k2 = (last_values_2y + last_values_y) / (2);
    // printf("%f %f %f %f", h1, k1, h2, k2);
    // archivoseeds << h1 << ' ' << k1 << endl;
    // archivoseeds << h2 << ' ' << k2 << endl;
    // archivoseeds.close();
    gnuplot p;
    string styleplot = "";
    /**
     * @brief en este for se generan los estilos para cada trayectoria
     */
#pragma omp parallel for
    for (int i = 0; i < maxParticles; i++)
    {
        //"linetype " + to_string(i + 1) + " linewidth 1 \\\n" +
        styleplot = "set style line " + to_string(i + 1) + " \\\n" +
                    "linecolor " + to_string(i + 1) + " \\\n" +
                    "linetype " + to_string(0) + " linewidth 1 \\\n" +
                    "pointtype " + to_string(i + 1) + " pointsize 1";

        p(styleplot);
    }

    /**
     * @brief en este for se crea la orden para que se grafiquen todos los puntos en una misma grafica
     */
    string plotline = "plot ";

    for (int i = 0; i < maxParticles; i++)
    {
        string filename = "'datos" + to_string(i) + ".dat'";
        plotline = plotline + "'datos" + to_string(i) + ".dat' with linespoints linestyle " + to_string(i + 1) + " title '',\\\n";
    }
    // plotline = plotline + "'seeds.dat' with linespoints linewidth 4 title ''";
    p(plotline);

    /**
     * @brief hacemos esperar el programa para que gnuplot alcance a leer los archivos
     *
     */
    sleep(5);

// std::cout << "\n\n\nEspere el grafico y luego de enter para eliminar los archivos .dat\n\n\n";
// std::cin.ignore();

/**
 * @brief for para eliminar archivos generado
 *
 */
#pragma omp parallel for
    for (int i = 0; i < maxParticles; i++)
    {
        string filename = "datos" + to_string(i) + ".dat";
        remove(filename.c_str());
    }
}