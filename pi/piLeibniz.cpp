#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <chrono>
using namespace std::chrono;
using namespace std;

/**
 * g++ -o piLeibniz  piLeibniz.cpp -fopenmp -w
 *
 * @param iteraciones
 * @return double
 */
double calcularPi(long int iteraciones)
{   
    double pi = 0;    

    auto start = high_resolution_clock::now();
    double start2 = omp_get_wtime();

    for (int x=0; x < iteraciones; x++){
        pi += pow(-1,x) / (2*x + 1);
    }

    double stop2 = omp_get_wtime();
    auto stop = high_resolution_clock::now();
    
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Tiempo gastado en la ejecución: " << duration.count() << " microseconds" << endl;
    cout << "Tiempo gastado en la ejecución (función de omp): " << (stop2 - start2) * 1000000 << " microseconds" << endl;

    return pi*4;
}

int main()
{
    double iteraciones = 1000000000;
    double pi;

    pi = calcularPi(iteraciones);
    cout << "Pi: " << pi << endl; 
    
}
