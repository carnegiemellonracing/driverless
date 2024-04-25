//
// Created by anthony on 4/24/24.
//

#ifndef DRIVERLESS_TEST_H
#define DRIVERLESS_TEST_H

/** .hpp */

class Base {
    Base (int x);
};

/** .cuh */
class Derived : public Base {
    Derived (int x);
    int m_x;
};

/** .cu */
Derived::Derived (int x) : m_x {x} {}
Base::Base (int x) {}


#endif //DRIVERLESS_TEST_H
