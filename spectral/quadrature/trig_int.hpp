#pragma once

#include <list>

#include "enum/enum.hpp"


namespace boltzmann {

/**
 * @brief integrates a product of sin/cos(l x) sin/cos(l1 x) ... sin/cos(ln x)
 *        over [-pi, pi]
 *        applies recursively the product rule for sin, cos
 *
 * @param t     leading type, 'c' or 's' (sin|cos)
 * @param l     leading frequency
 * @param tlist std::list( {'c' | 's'})
 * @param llist containing the frequencies
 *
 * @return
 */
double
trig_int(enum TRIG t, int l, std::list<enum TRIG> tlist, std::list<int> llist);

}  // end namespace boltzmann
