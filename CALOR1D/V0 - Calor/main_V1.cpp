#include "iostream"
#include <math.h>
#include <vector>
#include <iomanip>
#include <string>
#include "gnuplot.h"
#include <fstream>
using namespace std;

int main(int argc, char **argv)
{    
    string nombreArchivo = "datos.dat";
    ofstream archivo;
    double alfa = 1;
    float L = 1;
    float tmax = 1;
    double dx = 0.25;
    double dt = 0.02;
    double dtcalc;
    double CFL = 0.16;
    double T_inicial = 20;
    double T_externa = 100;

    cout << "Ingrese dt: ";
    cin >> dt;
    cout << "Ingrese dx: ";
    cin >> dx;
    cout << "Ingrese CFL: ";
    cin >> CFL;

    int Nx = (L / dx) + 1;

    vector<double> X;
    for (int i = 0; i < Nx; i++) //creacion del vector de longitudes espaciado
    {
        X.push_back(i * dx);
    }

    const int Nt = ceil(tmax / dt) + 1; //cantidad de tiempos
    vector<double> T;                   //matriz de tiempos (v) y posiciones (>)
    for (int z = 0; z < Nt * Nx; z++)
    {
        T.push_back(0);
    }
    for (int x = 0; x < Nx; x++) //seteo de la primera fila con temperatura ambiente de la barra
    {
        T[0 + x] = T_inicial;
    }
    for (int t = 1; t < Nt; t++)
    {
        T[(t)*Nx] = T_externa;           //Seteo de la primera columna con el valor del calor
        T[(t + 1) * Nx - 1] = T_externa; //seteo de la ultima columna con el valor del calor
    }
    T[0] = (T[1] + T[Nx]) / 2;                   //seteo del primer elemento
    T[Nx - 1] = (T[Nx - 1] + T[2 * Nx - 1]) / 2; //seteo del ultimo elemento primera fila
    for (int f = 1; f < Nt; f++)                 //f -> tiempo t
    {
        for (int c = 1; c < Nx - 1; c++) //c -> posicion x
        {
            T[f * Nx + c] = T[(f - 1) * Nx + c] + CFL * (T[(f - 1) * Nx + c + 1] - 2 * T[(f - 1) * Nx + c] + T[(f - 1) * Nx + c - 1]);
        }
    }
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
            if(tiempo.size()==0){
                tiempo.push_back(0);
            }
            cout << endl;
            cout << to_string(((i + 1) * dt) / Nx) << " | ";
            tiempo.push_back(((i + 1) * dt) / Nx);
        }
    }

    cout << endl;

    for (int i = 0; i <= (Nt)-2; i++){
        for (int j = 0; j < Nx; j++)
        {
            X.push_back(j * dx);
        }
    }

    vector<float> t;
    for (int j = 0; j < Nt; j++)
    {   
        for (int i = 0; i < Nx; i++){
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
    
    // p("unset label");
    // p("unset grid");
    // p("set terminal wxt size 700,400 font 'Verdana,10'"); 
    // p("set size ratio -1");
    //p("set palette rgbformulae 22,13,-31");
    //p("set xtics 0.2");
    //p("set ytics 0.2");
    p("set view map");
    p("set dgrid3d");
    p("set pm3d interpolate 0,0");
    p("splot 'datos.dat' using 1:2:3 with pm3d");

}
