#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <stdlib.h>
using namespace std;

int main()
{
    // suponemos dt
    float alfa = 1, dx = 0.25, L = 1, tmax = 1, dt = 0.03, CFL, dtcalculado, Nx, Nt;
    vector<float> x;
    vector<vector<float>> T;

A:
    system("CLS");
    cout << "Ingrese dt: ";
    cin >> dt;
    //CFL = (alfa * dt) / (pow(dx, 2));
    CFL = 0.16; // lambda
    // if (CFL < 0.5){
    //     printf("dt=%f es aceptable \n",dt);
    // }else{
    //     printf("dt=%f no es aceptable\n",dt);
    //     // goto A;
    // }
    // dt = pow(dt,2)/(2*alfa);

    for (float i = 0; i <= L; i = i + dx)
    {
        x.push_back(i);
    }

    Nt = ceil(tmax / dt) + 1;
    Nx = x.size();

    // CREAR LA MATRIZ LLENA DE CEROS
    T.resize(Nt, vector<float>(Nx, 0));

    for (int f = 0; f < T.size(); f++)
    {
        for (int c = 0; c < T[f].size(); c++)
        {
            // CONDICIONES FRONTERA
            // llena la primer fila
            if (f == 0)
            {
                T[0][c] = 20;
            }

            // llena la primer columna
            if (c == 0)
            {
                T[f][0] = 100;
            }

            // llena la ultima columna
            if (c == T[0].size() - 1)
            {
                T[f][c] = 100;
            }
        }
    }

    // ESQUINAS
    T[0][0] = (T[0][1] + T[1][0]) / 2;
    T[0][T[0].size() - 1] = (T[0][1] + T[1][0]) / 2;

    // APLICAR ECUACION DEL CALOR 1D
    for (int f = 1; f < T.size() - 1; f++)
    {
        for (int c = 1; c < T[f].size() - 1; c++)
        {
            T[f][c] = T[f - 1][c] + CFL * (T[f - 1][c + 1] - 2 * T[f - 1][c] + T[f - 1][c - 1]);
        }
    }

    for (int f = 0; f < T.size(); f++)
    {
        cout << "[ ";
        for (int c = 0; c < T[f].size(); c++)
        {
            if (f == 0 && (c == 0 || c == 4))
            {
                cout << to_string(T[f][c]) + "0"
                     << " ";
            }
            else if (f == T.size() - 1 && (c > 0 && c < 4))
            {
                cout << to_string(T[f][c]) + "0"
                     << " ";
            }
            else
            {
                cout << to_string(T[f][c]) << " ";
            }
        }
        cout << "]" << endl;
    }

    // cout << T.size() << " " << T[0].size() << endl;

    system("PAUSE()");

    goto A;
}