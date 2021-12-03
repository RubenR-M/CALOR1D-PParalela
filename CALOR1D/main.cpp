#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include "gnuplot.h"
#include <fstream>
#include <sstream>
using namespace std;

int main()
{
    string nombreArchivo = "datos.dat";
    string strT_0;
    ofstream archivo;
    double L, tmax, k;
    int Nt, Nx;
    vector<double> T_0;
    double dx, dt, T_izq, T_der;

    string path = "entradas.txt";          // Storing your filename in a string
    ifstream fin;                          // Declaring an input stream object

    fin.open(path);                        // Open the file
    if(fin.is_open())                      // If it opened successfully
    {
        fin >> Nt >> Nx >> dt >> dx >> strT_0 >> T_izq >> T_der >> k;  // Read the values and
        // store them in these variables
        fin.close();                   // Close the file
    }
    // cout<<"Nt: "<<Nt<<endl;
    // cout<<"Nx: "<<Nx<<endl;
    // cout<<"dt: "<<dt<<endl;
    // cout<<"dx: "<<dx<<endl;
    // cout<<"T_izq: "<<T_izq<<endl;
    // cout<<"T_der: "<<T_der<<endl;
    // cout<<"k: "<<k<<endl;
    while(1) //Use a while loop, "i" isn't doing anything for you
    {
        //if comman not found find return string::npos

        if (strT_0.find(',')!=std::string::npos)
        {
            double value;
            istringstream (strT_0) >> value;

            T_0.push_back(value);

            //Erase all element including comma
            strT_0.erase(0, strT_0.find(',')+1);
        }
        else
            break; //Come out of loop
    }
    cout<<"str: "<<strT_0<<endl;
    cout<<"T_0"<<endl;
    for (double i:T_0)
    {
        cout<<i<<" ";
    }

    L = Nx*dx;
    tmax=Nt*dt;

     Nx = (L / dx) + 1;

    vector<double> X;
    for (int i = 0; i < Nx; i++) //creacion del vector de longitudes espaciado
    {
        X.push_back(i * dx);
    }

//    const int Nt = ceil(tmax / dt) + 1; //cantidad de tiempos
    vector<double> T;                   //matriz de tiempos (v) y posiciones (>)
    for (int z = 0; z < Nt * Nx; z++)
    {
        T.push_back(0);
    }
    for (int x = 1; x < Nx-1; x++) //seteo de la primera fila con temperatura ambiente de la barra
    {
        T[x] = T_0[x-1];
    }
    for (int t = 1; t < Nt; t++)
    {
        T[(t)*Nx] = T_izq;           //Seteo de la primera columna con el valor del calor
        T[(t + 1) * Nx - 1] = T_der; //seteo de la ultima columna con el valor del calor
    }
    T[0] = (T[1] + T[Nx]) / 2;                   //seteo del primer elemento
    cout<<endl<<T[1]<<" "<<T[Nx]<<": "<<T[0]<<endl;
    T[Nx - 1] = (T[Nx - 2] + T[2 * Nx - 1]) / 2; //seteo del ultimo elemento primera fila
    cout<<T[Nx - 2]<<" "<<T[2 * Nx - 1]<<": "<<T[Nx - 1]<<endl;
    for (int f = 1; f < Nt; f++)                 //f -> tiempo t
    {
        for (int c = 1; c < Nx - 1; c++) //c -> posicion x
        {
            T[f * Nx + c] = T[(f - 1) * Nx + c] + k * (T[(f - 1) * Nx + c + 1] - 2 * T[(f - 1) * Nx + c] + T[(f - 1) * Nx + c - 1]);
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
    p("set view map");
    p("set dgrid3d");
    p("set pm3d interpolate 0,0");
    p("splot 'datos.dat' using 1:2:3 with pm3d");

}
