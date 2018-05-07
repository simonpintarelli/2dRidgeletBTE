#include "trig_int.hpp"

#include <iostream>

#define PI 3.141592653589793238462643383279502884197

using namespace std;
namespace boltzmann {

double trig_int(enum TRIG t, int l, list<enum TRIG> tlist, list<int> llist)
{
  if (tlist.size() == 0) {
    if (t == COS && l == 0)
      return 2 * PI;
    else
      return 0;
  } else {
    char tnext = tlist.front();
    tlist.pop_front();
    int lnext = llist.front();
    llist.pop_front();

    if (t == SIN && tnext == SIN)
      return 0.5 * trig_int(COS, l - lnext, tlist, llist) -
             0.5 * trig_int(COS, l + lnext, tlist, llist);
    else if (t == COS && tnext == COS)
      return 0.5 * trig_int(COS, l - lnext, tlist, llist) +
             0.5 * trig_int(COS, l + lnext, tlist, llist);
    else if (t == SIN && tnext == COS)
      return 0.5 * trig_int(SIN, l - lnext, tlist, llist) +
             0.5 * trig_int(SIN, l + lnext, tlist, llist);
    else if (t == COS && tnext == SIN)
      return 0.5 * trig_int(SIN, lnext - l, tlist, llist) +
             0.5 * trig_int(SIN, l + lnext, tlist, llist);
  }
  std::cerr << "uncaught state in trig_int!!!\n";
  throw 0;
}

}  // end namespace boltzmann
