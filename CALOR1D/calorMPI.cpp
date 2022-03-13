#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <mpi.h>
using namespace std;

// Envio de archivos a Guane
// scp ./calorMPI.cpp yrgualdronh@167.249.40.26:~
// scp ./entradas.txt yrgualdronh@167.249.40.26:~

// Carga, compilacion y ejecución en Guane
// module load devtools/mpi/openmpi/3.1.4 && module load devtools/gcc/9.2.0
// mpicxx calorMPI.cpp -o calorMPI
// sbatch ./run_mpi.sh

// Graficacion en local
// scp yrgualdronh@167.249.40.26:datos.dat ~/Downloads && ./graph datos.dat

/**
 * @brief valores por defecto si no existe el archivo de entradas
 */
int Nt = 10;
int Nx = 7;
double dt = 0.01;
double dx = 0.1;
double T_0[] = {20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 20.0};
double T_izq = 50, T_der = 50;
double k = 0.16;

/**
 * @brief Esta funcion sigue la formula de la ecuacion para hallar cada valor. Donde recibe un vector con lo valores
 * necesarios para calcular la posicion deseada y numero de calculos para el core.
 *
 * @param nx -> cantidad de valores a calcular
 * @param rT -> valores en el tiempo (t-1) para calcular los valores del tiempo t de tamaño nx+2
 * @return vector<double>  nT -> vector con los nx valores calculados
 */
vector<double> calc(int nx, vector<double> &rT)
{
    vector<double> nT;
    for (int i = 0; i < nx; i++)
    {
        nT.push_back(rT[i + 1] + k * (rT[i + 2] - 2 * rT[i + 1] + rT[i]));
    }
    return nT;
}

int main(int argc, char **argv)
{
    string nombreArchivo = "datos.dat";
    string strT_0;
    ofstream archivo;
    vector<double> T_0; // vector de temperaturas iniciales
    /**
     * @brief lo que hace el siguiente bloque de codigo es leer el archivo llamado entradas.txt
     * y asignar cada valor correspondiente
     */
    string path = "entradas.txt"; // nombre del archivo con condiciones iniciales
    ifstream fin;
    fin.open(path);
    if (fin)
    {
        // Declaring an input stream object
        // Open the file
        if (fin.is_open()) // If it opened successfully
        {
            fin >> Nt >> Nx >> dt >> dx >> strT_0 >> T_izq >> T_der >> k; // Read the values and store them in these variables
            fin.close();                                                  // Close the file
        }
        while (1) // Use a while loop, "i" isn't doing anything for you
        {
            if (strT_0.find(',') != std::string::npos) // if comman not found find return string::npos
            {
                double value;
                istringstream(strT_0) >> value;
                T_0.push_back(value);
                strT_0.erase(0, strT_0.find(',') + 1); // Erase all element including comma
            }
            else
                break; // Come out of loop
        }
    }
    else
    {
        cout << "No se encuentra el archivo < entradas.txt >, se usaran valores por defecto." << endl;
    }

    /**
     * @brief los siguientes for's realizan el llenado de las temperaturas
     * inicialmente lleno de ceros pero luego basados en las condiciones deseadas
     * se llenan la base y los laterales de la barra.
     */
    Nx = Nx + 1;
    vector<double> X; // vector de posiciones
    vector<double> T; // vector de las temperaturas
    for (int z = 0; z < Nt * Nx; z++)
    {
        T.push_back(0);
    }
    for (int i = 0; i < Nx; i++)
    {
        X.push_back(i * dx); // creacion del vector de longitudes espaciado
        T[i] = T_0[i];       // seteo de la primera fila con temperatura ambiente de la barra
    }
    for (int t = 1; t < Nt; t++)
    {
        T[(t)*Nx] = T_izq;           // Seteo de la primera columna con el valor del calor
        T[(t + 1) * Nx - 1] = T_der; // seteo de la ultima columna con el valor del calor
    }

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // dado la logica usada para resolver el problema
    // se encontro que se deben considerar las siguientes condiciones
    // para el buen funcionamiento del programas
    if (((Nx - 2) % (size - 1) != 0 || size == 1) && rank == 0)
    {
        cout << "\n\nNo es posible segmentar el cálculo.\n";
        printf("Intenta usar una cantidad de nodos de cálculo (np-1) que sea divisor de Nx-2=%d.\n\n\n", Nx - 2);
    }
    else
    {
        // numero de elementos correspondiente para cada core
        int nx = (int)((Nx - 2) / (size - 1)); // CORRECIONES: Se divide entre la cantidad de ranks de calculo y se descartan los extremos

        vector<double> rT; // vector usado para guardar los calculos
        for (int z = 0; z < nx + 2; z++)
        {
            rT.push_back(0);
        }

        /**
         * @brief por cada paso de tiempo cada core tomara una parte de la fila a calcular
         */
        for (int t = 0; t < Nt; t++)
        {
            // el core 0 sera el que enviara los datos de los tiempos t y recibira los valores calculados por los demas cores
            if (rank == 0)
            {
                for (int r = 1; r < size; r++) // CORRECIONES: r=0 -> r=1
                {
                    // envia a cada core de calculo la seccion de valores (nx+2) del tiempo t
                    // necesaria para el calculo de su correspondientes (nx) valores en t+1
                    MPI_Send(&T[t * Nx + (r - 1) * nx], nx + 2, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);

                    // recibe los calculos realizados por cada core y los asigna directamente al vector de temperaturas
                    MPI_Recv(&T[(t + 1) * Nx + (r - 1) * nx + 1], nx, MPI_DOUBLE, r, 1, MPI_COMM_WORLD, &status);
                }
            }
            // los demas cores realizaran los calculos
            else
            {
                // se recibe los nx+2 valores del tiempo t para calcular los nx del tiempo t+1
                MPI_Recv(&rT[0], nx + 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

                // se hace un llamado a la funcion para calcular los nx valores de la seccion en el tiempo t+1
                vector<double> nT = calc(nx, rT);

                // se envian los valores ya calculados
                MPI_Send(&nT[0], nx, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            }
        }
        MPI_Finalize();

        /**
         * @brief El siguiente bloque de codigo imprime la matriz de los valores ya calculados
         * adicionalmente llena los vectores de posiciones y tiempos necesarios para luego pasar
         * esos valores a un archivo que puede ser usado en gnuplot para ver el grafico
         */
        if (rank == 0)
        {
            cout << "\n\n";
            cout << fixed << setprecision(1);
            cout << "t\\x    | ";
            for (int i = 0; i < Nx; i++) // creacion del vector de longitudes espaciado
            {
                printf("%.1f00 ", X[i]);
            }
            cout << endl;
            cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;

            vector<float> tiempo;
            vector<float> t;
            double ti;

            // Mostrar
            for (int i = 0; i < Nx * Nt; i++)
            {
                if (i == 0)
                {
                    printf("%.1f000 | ", i * dt);
                }
                if (i == 0 || i == Nx - 1)
                {
                    printf("%.1f0 ", T[i]);
                }
                else if (to_string(T[i]).size() < 12)
                {
                    if (T[i] < 10)
                    {
                        printf("%.1f00 ", T[i]);
                    }
                    else if (T[i] < 100)
                    {
                        printf("%.1f0 ", T[i]);
                    }
                    else
                    {
                        printf("%.1f ", T[i]);
                    }
                }

                if (i != Nx * Nt - 1 && (i + 1) % Nx == 0)
                {
                    if (tiempo.size() == 0)
                    {
                        tiempo.push_back(0);
                    }
                    ti = ((i + 1) * dt) / Nx;
                    if (ti < 100)
                    {
                        printf("\n%.1f00 | ", ti);
                    }
                    else if (ti < 1000)
                    {
                        printf("\n%.1f0 | ", ti);
                    }
                    else
                    {
                        printf("\n%.1f | ", ti);
                    }
                    tiempo.push_back(ti);
                }
            }

            // llenado de posiciones
            for (int i = 0; i < (Nt)-1; i++)
            {
                for (int j = 0; j < Nx; j++)
                {
                    X.push_back(j * dx);
                }
            }
            // llenado de tiempos
            for (int j = 0; j < Nt; j++)
            {
                for (int i = 0; i < Nx; i++)
                {
                    t.push_back(tiempo[j]);
                }
            }

            // escritura del archivo donde se ponen POSICION TIEMPO TEMPERATURA
            archivo.open(nombreArchivo.c_str(), fstream::out);
            for (int i = 0; i < X.size(); i++)
            {
                archivo << X[i] << " " << t[i] << " " << T[i] << endl;
            }
            archivo.close();
            cout << "\n\n";
        }
    }
}