#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <fstream>
using namespace std;

int main(int argc, char **argv)
{
    // suponemos dt
    float alfa = 1, dx = 0.25, L = 1, tmax = 1, dt = 0.03, CFL, dtcalculado, Nx, Nt;
    int k;
    vector<float> x;
    vector<vector<float>> T;
    //vector<float> T2;

    // Create a text string, which is used to output the text file
    string myText;

    // Read from the text file
    ifstream MyReadFile(argv[1]);

    // Use a while loop together with the getline() function to read the file line by line
    while (getline(MyReadFile, myText))
    {
        // Output the text from the file
        cout << myText;
    }

    // Close the file
    MyReadFile.close();

    //cout << argv[1] << endl;

    CFL = 0.16; // lambda

    for (float i = 0; i <= L; i = i + dx)
    {
        x.push_back(i);
    }

    Nt = ceil(tmax / dt) + 1;
    Nx = x.size();
    //cout << Nt <<" "<< Nx<< endl;
    // CREAR LA MATRIZ LLENA DE CEROS
    T.resize(Nt, vector<float>(Nx, 0));

    //for (int i = 0; i < T.size(); i++)
    //{
    //    for (int j = 0; j < T[i].size(); j++)
    //    {
    //        T2.push_back(T[i][j]);
    //    }
    //
    //}
}