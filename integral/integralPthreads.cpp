#include <iostream>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <pthread.h>
using namespace std;
using namespace std::chrono;

/**
 * Compilacion:
 *  -> g++ integralOpenMP.cpp -o integral -fopenmp -pthread
 * Ejecucion:
 *  -> ./integral $ITERATION $a $b $NUM_THREADS
 *  -> ./integral 100000000 0 3.141516 4
 */

static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
int num_threads;
long int iteraciones;
double area = 0.0;
double a,b; 

void *integral(void *identificador)
{
    double param, delta, k;
    double integralThread = 0;

    long int iteracionesXhilo = iteraciones / num_threads;
    long int inicioXhilo = (long)identificador * iteracionesXhilo;
    long int finxHilo = ((long)identificador + 1) * iteracionesXhilo;

    delta = (b - a) / iteraciones;
    if ((long)identificador == 0)
    {
        printf("hilo:%li \titeracionesXhilo:%d \tinicioXhilo:%d \t\tfinxHilo:%d\n", (long)identificador, iteracionesXhilo, inicioXhilo, finxHilo);
    }else{
        printf("hilo:%li \titeracionesXhilo:%d \tinicioXhilo:%d \tfinxHilo:%d\n", (long)identificador, iteracionesXhilo, inicioXhilo, finxHilo);
    }
    
    
    for (int i = inicioXhilo; i <= finxHilo; i++)
    {        
        k = i - 1 / 2;
        param = a + (k)*(delta);
        integralThread = integralThread + sin(param) * delta;
    }

    area += integralThread;
}

int main(int argc, char **argv)
{
    num_threads = atoi(argv[4]);

    pthread_t threads[num_threads];
    int integralThreads;

    char *stopstring;
    
    iteraciones = strtod(argv[1], &stopstring);
    a = strtod(argv[2], &stopstring);
    b = strtod(argv[3], &stopstring);

    auto start = high_resolution_clock::now();
    double start2 = omp_get_wtime();

    for (int i = 0; i < num_threads; i++)
    {
        integralThreads = pthread_create(&threads[i], NULL, integral, (void *)i);
    }

    for (int i = 0; i < num_threads; i++)
    {
        integralThreads = pthread_join(threads[i], NULL);
    }

    double stop2 = omp_get_wtime();
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    printf("Area: %f\n", area);
    cout << "Tiempo gastado en la ejecución (funcion de crono): " << (double)duration.count() / 1000000 << " segundos" << endl;
    cout << "Tiempo gastado en la ejecución (función de omp  ): " << (stop2 - start2) << " segundos" << endl;
    return 0;
}