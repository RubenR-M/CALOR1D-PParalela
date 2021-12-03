#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <math.h>
using namespace std;

int main()
{   
    float alfa=1, dx=0.25, L =1, tmax=1, dt=0.02, CFL;
    vector<int> prueba;

    CFL = (alfa*dt)/(pow(dx,2));
    // prueba.push_back(60);
    // prueba.push_back(12);
    // prueba.push_back(54);
    cout << CFL << endl;
    
    if (CFL < 1/2){
        printf("dt=%f es aceptable\n",dt);
    }else{
        printf("dt=%f no es aceptable\n",dt);
    }

    // for (int i = 0; i < prueba.size(); i++){
    //     cout << i << endl;
    // }    
    
    // //para este momento, el vector tiene 60,12,54
    // for (int i: prueba) {
    //     cout << i << ' '<< endl; // will print: "a b c"
    // }
    // cout << "--------------------------------"<< endl;
    // sort(prueba.begin(), prueba.end()); //listo, array ordenado, ahora tiene 12,54,60
    // for (int i: prueba) {
    //     cout << i << ' '<< endl; // will print: "a b c"
    // }
    return 0;
}