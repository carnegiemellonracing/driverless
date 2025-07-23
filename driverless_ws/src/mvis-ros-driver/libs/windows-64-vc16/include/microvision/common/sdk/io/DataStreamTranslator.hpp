//==============================================================================
//!\file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 26, 2020
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <functional>
#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Template to implement translators.
//!
//! Implement this interface to apply the common translator template.
//------------------------------------------------------------------------------
template<typename InputType, typename OutputType>
class DataStreamTranslator
{
public:
    //========================================
    //! \brief Definition for input type.
    //----------------------------------------
    using Input = InputType;

    //========================================
    //! \brief Definition for output type.
    //----------------------------------------
    using Output = OutputType;

    //========================================
    //! \brief Shared pointer for input type.
    //----------------------------------------
    using InputPtr = std::shared_ptr<Input>;

    //========================================
    //! \brief Shared pointer for output type.
    //----------------------------------------
    using OutputPtr = std::shared_ptr<Output>;

    //========================================
    //! \brief Callback function for output.
    //----------------------------------------
    using OutputCallback = std::function<void(const OutputPtr&)>;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~DataStreamTranslator() = default;

public:
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    virtual void setOutputCallback(const OutputCallback& callback) = 0;

    //========================================
    //! \brief Translate input to output and call callback if output is complete.
    //! \param[in] input  Input to process
    //! \returns Either \c true if input is valid, otherwise \c false.
    //----------------------------------------
    virtual bool translate(const InputPtr& input) = 0;

    //========================================
    //! \brief Clean up the translator state.
    //----------------------------------------
    virtual void clear() = 0;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
