//==============================================================================
//! \file
//!
//! \brief Contains helper functions for converting to SI units.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 2nd, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/misc/units/Unit.hpp>

//==============================================================================

//! \page page_si_units SI UNITS
//! This is the current list of SI units supported by the SDK. Not all conversions are available.
//! If required further \ref unit::Convert specializations can be implemented in application code.
//! To add your own custom unit types, have a look at example in the unit tests Unit<length::testmeter>
//! in UnitTests.cpp.
//!
//! \li time: second
//! \li length: meter
//! \li mass: kilogram
//! \li temperature: kelvin
//! \li angle: radian
//! \li frequency: hertz
//! \li velocity: meter per second
//! \li angular velocity: radian per second
//! \li acceleration: meter per second^2
//! \li force: newton
//! \li power: watt
//! \li torque: newton meter
//! \li area: m^2

// Not yet supported si units:
//
// current: ampere
// amount: mole
// luminosity: candela
// solid angle: steradian
// pressure: pascal
// charge: coulomb
// energy: joule
// voltage: volt
// capacitance: farad
// impedance: ohm
// conductance: siemens
// magnetic flux: weber
// magnetic field strength: tesla
// inductance: henry
// luminous flux: lumen
// illuminance: lux
// radiation source activity: bequerel
// absorbed radiation dose: gray
// equivalent radiation dose: sievert
// volume: m^3
// density: kilogram per m^3
// concentration: parts per million

#include <microvision/common/sdk/misc/units/time.hpp>
#include <microvision/common/sdk/misc/units/length.hpp>
#include <microvision/common/sdk/misc/units/mass.hpp>
#include <microvision/common/sdk/misc/units/temperature.hpp>
#include <microvision/common/sdk/misc/units/angle.hpp>
#include <microvision/common/sdk/misc/units/frequency.hpp>
#include <microvision/common/sdk/misc/units/velocity.hpp>
#include <microvision/common/sdk/misc/units/angularvelocity.hpp>
#include <microvision/common/sdk/misc/units/power.hpp>
#include <microvision/common/sdk/misc/units/force.hpp>
#include <microvision/common/sdk/misc/units/acceleration.hpp>
#include <microvision/common/sdk/misc/units/torque.hpp>
#include <microvision/common/sdk/misc/units/area.hpp>

//==============================================================================
