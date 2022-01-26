#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <chrono>
using namespace std::chrono;
using namespace std;

/**
 * @brief 
 * 
 * @param iteraciones 
 * @return double 
 */
double calcularPi(long int iteraciones)
{
    int x; double pi = 0;
    int nproc = omp_get_num_procs();
    //int nproc = 1;
    omp_set_num_threads(nproc);

    auto start = high_resolution_clock::now();
    double start2 = omp_get_wtime();

#pragma omp parallel for reduction(+:pi)
    for (x=0; x < iteraciones; x++){
        pi += pow(-1,x) / (2*x + 1);
    }

    double stop2 = omp_get_wtime();
    auto stop = high_resolution_clock::now();
    
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Tiempo gastado en la ejecución: " << duration.count() << " microseconds" << endl;
    cout << "Tiempo gastado en la ejecución (función de omp): " << (stop2 - start2) * 1000000 << " microseconds" << endl;

    return pi*4;
}

/**
 * @brief 
 * 
 * @return int 
 */
int main()
{
    double iteraciones = 1000000000;
    double pi;

    pi = calcularPi(iteraciones);
    cout << "Pi: " << pi << endl; 
    
}
