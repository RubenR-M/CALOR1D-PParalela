#include <iostream>
#include <math.h>
#include <string.h>
#include <omp.h>
using namespace std;

/**
 * Compilacion:
 *  -> g++ int_rect.cpp -o int_rect -fopenmp
 * Ejecucion:
 *  -> ./int_rect $a $b $ITERATION $NUM_THREADS
 *  -> ./int_rect 0 3.141516 100000000 4
 */

/**
 * Este codigo tiene un problema desconocido
 * ya que usa un solo hilo 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char* argv[])
{

    omp_set_num_threads(stoi(argv[4]));
    
    double start = omp_get_wtime();
    double a = stof(argv[1]);
    double b = stof(argv[2]);
    int n = stoi(argv[3]);
    double delta = (b-a)/n;
    double result=0;
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
#pragma omp paralell for private(i) reduction(+:result)
    for (int i=1; i<=n; i++)
    {
        result += sin(a+(i-1/2)*delta)*delta;
    }
    
    cout<<"\tResult: "<<result<<"\n";
    double stop = omp_get_wtime();
    printf("\tTiempo 'MAIN' (omp): %f seconds. \n", (stop - start) );
}