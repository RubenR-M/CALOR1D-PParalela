#include <math.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <pthread.h>
using namespace std::chrono;
using namespace std;

/**
 * Compilacion:
 *  -> g++ -o piLeibniz_threads  piLeibniz_threads.cpp -pthread -fopenmp -w
 *
 * Guias:
 *  -> https://www.youtube.com/watch?v=QMNtAFZtFMA
 *  -> https://www.youtube.com/watch?v=pLa972Rgl1I&list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG&index=9
 *  -> https://www.tutorialspoint.com/cplusplus/cpp_multithreading.htm
 */

static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
int num_threads = 4;
long int iteraciones = 1000000000000;
double pi = 0.0;

void *calcularPi(void *identificador)
{
    double piThread = 0;
    int signo = 1;

    /**
     * Como en OpenMp cada hilo tomara un grupo de iteraciones
     * por lo tanto se dividen las iteracion por el numero de hilos
     * para definir ese grupo de iteraciones se multiplica por el id del hilo para 
     * el inicio de la iteracion y para su fin seria el inicio del siguiente grupo de iteraciones
     * del proximo thread
     */
    int iteracionesXhilo = iteraciones / num_threads;
    int inicioXhilo = (long)identificador * iteracionesXhilo;
    int finxHilo = ((long)identificador + 1) * iteracionesXhilo;

    for (int i = inicioXhilo; i < finxHilo; i++)
    {
        piThread += signo * (1.0 / (2 * i + 1));
        signo *= -1;
    }

    /**
     *  si no me equivo estas funciones indican una parte critica del codigo
     *  como en el video tutorial de OpenMP esta seria una seccion critica
     */
    pthread_mutex_lock(&m);
    pi += piThread;
    pthread_mutex_unlock(&m);
    return 0;
}

int main()
{
    pthread_t threads[num_threads];
    int piThreads;

    auto start = high_resolution_clock::now();
    double start2 = omp_get_wtime();
    
    /**
     * se crea cada hilo
     *  -> se le indica la funcion
     *  -> y como parametro el identificador del thread
     */
    for (int i = 0; i < num_threads; i++)
    {
        piThreads = pthread_create(&threads[i], NULL, calcularPi, (void *)i);
    }

    /**
     * La subrutina pthread_join() bloquea el subproceso de llamada hasta que 
     * finaliza el subproceso 'threadid' especificado. Cuando se crea un subproceso, 
     * uno de sus atributos define si se puede unir o separar. Solo se pueden unir 
     * los subprocesos que se crean como unibles. Si un subproceso se crea como 
     * separado, nunca se puede unir.
     */
    for (int i = 0; i < num_threads; i++)
    {
        piThreads = pthread_join(threads[i], NULL);
    }

    double stop2 = omp_get_wtime();
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Pi: " << pi*4 << endl;
    cout << "Tiempo gastado en la ejecución (funcion de crono): " << (double) duration.count() / 1000000 << " segundos" << endl;
    cout << "Tiempo gastado en la ejecución (función de omp  ): " << (stop2 - start2)<< " segundos" << endl;
    return 0;
}
