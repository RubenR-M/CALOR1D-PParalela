#include <math.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
using namespace std::chrono;
using namespace std;

int main()
{
    double pi = 0;
    int max=10000000;
    cout<<"Inserte el numero de iteraciones deseadas: ";
    cin>>max;
    
    auto start = high_resolution_clock::now();
    double start2 = omp_get_wtime();
#pragma omp parallel for reduction(+:pi)
    for (int n=0; n<max; n++)
    {
    	pi += pow(-1,n)/(2*n+1);
    }
    auto stop = high_resolution_clock::now();
    double stop2 = omp_get_wtime();
    auto duration = duration_cast<microseconds>(stop - start);
    cout<<"Tiempo gastado en la ejecución (chr): "<<duration.count()<<" microseconds" <<endl;
    cout<<"Tiempo gastado en la ejecución (omp): "<<(stop2-start2)*1000000<<" microseconds"<<endl;
    
    pi=4*pi;
    cout<<setprecision(10)<<fixed;
    cout<<pi<<endl;
}
