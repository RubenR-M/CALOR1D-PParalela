#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include "gnuplot.h"
#include <fstream>
#include <sstream>
#include <omp.h>
#include <chrono>
using namespace std;
// g++ main_OPENMP.cpp -o omp -fopenmp
// ./omp entradas.txt 8
int main(int argc, char **argv)
{
    auto start = chrono::high_resolution_clock::now();
    double start2 = omp_get_wtime();
    string nombreArchivo = "datos.dat";
    string strT_0;
    ofstream archivo;
    double L, tmax, k;
    int Nt, Nx;
    vector<double> T_0;
    double dx, dt, T_izq, T_der;

    string path = (string)argv[1]; // Storing your filename in a string
    omp_set_num_threads(stoi(argv[2]));
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
        if (z<Nx)
            X.push_back(z * dx); //creacion del vector de longitudes espaciado
    }
    
#pragma omp parallel for
    for (int i = 0; i < Nx; i++) 
    {
        T[i] = T_0[i];		//seteo de la primera fila con temperatura ambiente de la barra
    }

#pragma omp parallel for
    for (int t = 1; t < Nt; t++)
    {
        T[(t)*Nx] = T_izq;           //Seteo de la primera columna con el valor del calor
        T[(t + 1) * Nx - 1] = T_der; //seteo de la ultima columna con el valor del calor
    }

#pragma omp simd collapse(2)
    for (int f = 0; f < Nt-1; f++) //f -> tiempo t
    {
        for (int c = 1; c < Nx - 1; c++) //c -> posicion x
        {
            T[(f+1)*Nx+c]=T[f*Nx+c]+k*(T[f*Nx+c+1]-2*T[f*Nx+c]+T[f*Nx+c-1]);
        }
    }

    cout << setprecision(3) << fixed;
    cout << "t\\x      | ";
    for (int i = 0; i < Nx; i++)
    {
        if (i == 0 || i == Nx - 1)
        {
            cout << to_string(X[i]) << "00  ";
        }
        else
        {
            cout << to_string(X[i]) << "00  ";
        }
    }
    
    cout << endl << "---------------------------------------------------------------------------------------------------------------------------------------------" << endl;

    vector<float> tiempo;
    vector<float> t;

    //Mostrar
    for (int i = 0; i < Nx * Nt; i++)
    {
        if (i == 0)
        {
            cout << to_string(i * dt) << " | ";
        }
        if (i == 0 || i == Nx - 1)
        {
            cout << to_string(T[i]) << "0  ";
        }
        else if (to_string(T[i]).size() < 12)
        {
            if (T[i]<10)
            {
                cout<<"0"<<to_string(T[i]) << "0  ";
            }
            else
            {
                cout << to_string(T[i]) << "0  ";
            }
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
            cout<<endl;
            cout << to_string(((i + 1) * dt) / Nx) << " | ";
            tiempo.push_back(((i + 1) * dt) / Nx);
        }
    }

    //Graficar
    for (int i = 0; i < (Nt)-1; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            X.push_back(j * dx);
        }
    }
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
    cout<<endl;

    gnuplot p;
    p("set view map");
    p("set dgrid3d");
    p("set pm3d interpolate 0,0");
    p("splot 'datos.dat' using 1:2:3 with pm3d");
    auto stop = chrono::high_resolution_clock::now();
    double stop2 = omp_get_wtime();
    chrono::duration<double> elapsed_seconds = stop-start;
    cout <<endl<< "Tiempo (chr): " << elapsed_seconds.count() * 1000000 << " microseconds" << endl;
    cout << "Tiempo (omp): " << (stop2 - start2) * 1000000 << " microseconds" << endl<<endl;
}
