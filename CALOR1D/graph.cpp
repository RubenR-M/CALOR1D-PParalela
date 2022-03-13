#include "gnuplot.h"
int main(int argc, char **argv)
{
    gnuplot p;
    p("set view map");
    p("set dgrid3d");
    p("set pm3d interpolate 0,0");
    p("splot 'datos.dat' using 1:2:3 with pm3d");
}