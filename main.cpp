#include <iostream>
#include<time.h>
#include<stdlib.h>
using namespace std;

int main()
{
    srand(time(NULL));
    int a = rand();
    switch(a%4)
    {
        case 0:
            cout<<"red";
        break;
        case 1:
            cout<<"blue";
        break;
        case 2:
            cout<<"green";
        break;
        case 3:
            cout<<"black";
        break;

    }
    return 0;
}
