#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include "gnuplot.h"
#include <fstream>
#include <sstream>
#include <omp.h>
#include <chrono>
using namespace std::chrono;
using namespace std;

int main(int argc, char **argv)
{
    string nombreArchivo = "datos.dat";
    string strT_0;
    ofstream archivo;
    double L, tmax, k;
    int Nt, Nx;
    vector<double> T_0;
    double dx, dt, T_izq, T_der;

    string path = (string)argv[1]; // Storing your filename in a string
    ifstream fin;                 // Declaring an input stream object
    fin.open(path);    // Open the file
    if (fin.is_open()) // If it opened successfully
    {
        fin >> Nt >> Nx >> dt >> dx >> strT_0 >> T_izq >> T_der >> k; // Read the values and store them in these variables
        fin.close(); // Close the file
    }
    
    while (1) //Use a while loop, "i" isn't doing anything for you
    {
        if (strT_0.find(',') != std::string::npos) //if comman not found find return string::npos
        {
            double value;
            istringstream(strT_0) >> value;
            T_0.push_back(value);
            strT_0.erase(0, strT_0.find(',') + 1); //Erase all element including comma
        }
        else
            break; //Come out of loop
    }

    L = Nx * dx;
    Nx=Nx+1;
    tmax = Nt * dt;
    vector<double> X;
    vector<double> T; //"matriz" de tiempos (v) y posiciones (>)

    for (int z = 0; z < Nt * Nx; z++)
    {
        T.push_back(0);
    }
    
    auto start = high_resolution_clock::now();
    double start2 = omp_get_wtime();

#pragma omp parallel for
    for (int i = 0; i < Nx; i++) 
    {
        X.push_back(i * dx);	//creacion del vector de longitudes espaciado
        T[i] = T_0[i];		//seteo de la primera fila con temperatura ambiente de la barra
    }
    double stop2 = omp_get_wtime();
    auto stop = high_resolution_clock::now();

    // for(int n=0; n<size; ++n)
    // std::cout << sinTable[n] << " ";

    std::cout << std::endl;
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "Tiempo gastado en la ejecución: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Tiempo gastado en la ejecución (función de omp): " << (stop2 - start2) * 1000000 << " microseconds" << std::endl;

    start = high_resolution_clock::now();
    start2 = omp_get_wtime();

#pragma omp parallel for
    for (int t = 1; t < Nt; t++)
    {
        T[(t)*Nx] = T_izq;           //Seteo de la primera columna con el valor del calor
        T[(t + 1) * Nx - 1] = T_der; //seteo de la ultima columna con el valor del calor
    }
    stop2 = omp_get_wtime();
    stop = high_resolution_clock::now();

    // for(int n=0; n<size; ++n)
    // std::cout << sinTable[n] << " ";

    std::cout << std::endl;
    duration = duration_cast<microseconds>(stop - start);

    std::cout << "Tiempo gastado en la ejecución: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Tiempo gastado en la ejecución (función de omp): " << (stop2 - start2) * 1000000 << " microseconds" << std::endl;
    //T[0] = (T[1] + T[Nx]) / 2; //seteo del primer elemento
    //T[Nx - 1] = (T[Nx - 2] + T[2 * Nx - 1]) / 2; //seteo del ultimo elemento primera fila

    start = high_resolution_clock::now();
    start2 = omp_get_wtime();

#pragma omp parallel for
    for (int f = 1; f < Nt; f++) //f -> tiempo t
    {
        start = high_resolution_clock::now();
        start2 = omp_get_wtime();
#pragma omp parallel for
        for (int c = 1; c < Nx - 1; c++) //c -> posicion x
        {
            T[f * Nx + c] = T[(f - 1) * Nx + c] + k * (T[(f - 1) * Nx + c + 1] - 2 * T[(f - 1) * Nx + c] + T[(f - 1) * Nx + c - 1]);
        }
        stop2 = omp_get_wtime();
        stop = high_resolution_clock::now();

        // for(int n=0; n<size; ++n)
        // std::cout << sinTable[n] << " ";

        std::cout << std::endl;
        duration = duration_cast<microseconds>(stop - start);

        std::cout << "Tiempo gastado en la ejecución: " << duration.count() << " microseconds" << std::endl;
        std::cout << "Tiempo gastado en la ejecución (función de omp): " << (stop2 - start2) * 1000000 << " microseconds" << std::endl;
    }
    stop2 = omp_get_wtime();
    stop = high_resolution_clock::now();

    // for(int n=0; n<size; ++n)
    // std::cout << sinTable[n] << " ";

    std::cout << std::endl;
    duration = duration_cast<microseconds>(stop - start);

    std::cout << "Tiempo gastado en la ejecución: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Tiempo gastado en la ejecución (función de omp): " << (stop2 - start2) * 1000000 << " microseconds" << std::endl;

    cout << setprecision(3) << fixed;
    cout << "t/x      | ";
    for (int i = 0; i < Nx; i++) //creacion del vector de longitudes espaciado
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
            cout << to_string(T[i]) << "00  ";
        }
        else if (to_string(T[i]).size() < 12)
        {
            cout << to_string(T[i]) << "0  ";
        }
        else
        {
            cout << to_string(T[i]) << "  ";
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
        archivo << X[i] << " " << t[i] << " " << T[i] << endl;
    }
    archivo.close();

    gnuplot p;
    p("set view map");
    p("set dgrid3d");
    p("set pm3d interpolate 0,0");
    p("splot 'datos.dat' using 1:2:3 with pm3d");
}
