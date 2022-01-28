#include <iostream>
#include <math.h>
#include <omp.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

/**
 * Compilacion:
 *  -> g++ integralOpenMP.cpp -o integral -fopenmp
 * Ejecucion:
 *  -> ./integral $ITERATION $a $b $NUM_THREADS
 *  -> ./integral 100000000 0 3.141516 4
 */

/**
 * @param n
 * @param a
 * @param b
 * @return double
 */
double integral(int n, double a, double b){

    int i;
    double param, delta,k, result=0;

    /**
     * calculo de delta de X 
     */
    delta = (b - a) / n;
    /**
     * -> En OpenMp cada hilo tomara un grupo de iteraciones
     *    por lo tanto se dividen las iteracion por el numero de hilos
     * 
     * -> El prametro private indica que la variable iteradora del for es 
     *    diferente para cada hilo.
     * 
     * -> El parametro reduction tiene como funcion tomar la variable 
     *    y hacer privado su valor para cada hilo, de modo que al final
     *    de la ejecucion se sumen sus valores.
     */
#pragma omp parallel for private(i) reduction(+:result)
    for (i = 1; i <= n; i++)
    {   
        k = i - 1 / 2;
        param = a + (k)*(delta);
        result = result + sin(param)*delta;
    }
    
    return result;
}

/**
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char **argv)
{
    omp_set_num_threads(stoi(argv[4]));
    char *stopstring;
    double area;
    double n = strtod(argv[1], &stopstring);
    double a = strtod(argv[2], &stopstring);
    double b = strtod(argv[3], &stopstring);

    auto start = high_resolution_clock::now();
    double start2 = omp_get_wtime();

    area = integral(n,a,b);
    printf("Area: %f\n", area);

    double stop2 = omp_get_wtime();
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Tiempo gastado en la ejecución (funcion de crono): " << (double)duration.count() / 1000000 << " segundos" << endl;
    cout << "Tiempo gastado en la ejecución (función de omp  ): " << (stop2 - start2) << " segundos" << endl;
    return 0;
}