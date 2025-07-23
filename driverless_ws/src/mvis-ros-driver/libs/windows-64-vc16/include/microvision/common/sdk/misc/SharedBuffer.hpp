//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 8, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/Utils.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Shared pointer of selective binary buffer.
//!
//! The buffer is linked by shared buffer and selective by offset and size.
//------------------------------------------------------------------------------
class SharedBuffer final
{
public:
    //========================================
    //! \brief Wrapping buffer type.
    //----------------------------------------
    using BufferType = std::vector<char>;

    //========================================
    //! \brief Wrapping buffer pointer type.
    //----------------------------------------
    using BufferPointerType = std::shared_ptr<BufferType>;

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    SharedBuffer();

    //========================================
    //! \brief Wrapping buffer move constructor.
    //! \param[in] buffer  Wrapping buffer type instance.
    //----------------------------------------
    SharedBuffer(BufferType&& buffer);

    //========================================
    //! \brief Wrapping buffer copy constructor.
    //! \param[in] buffer  Wrapping buffer type instance.
    //----------------------------------------
    SharedBuffer(const BufferType& buffer);

    //========================================
    //! \brief Wrapping buffer pointer move constructor.
    //! \param[in] buffer  Wrapping buffer pointer type instance.
    //----------------------------------------
    SharedBuffer(BufferPointerType&& bufferPtr);

    //========================================
    //! \brief Wrapping buffer pointer copy constructor.
    //! \param[in] buffer  Wrapping buffer pointer type instance.
    //----------------------------------------
    SharedBuffer(const BufferPointerType& bufferPtr);

    //========================================
    //! \brief Wrapping buffer range constructor.
    //! \param[in] first    First iterator of range.
    //! \param[in] last     Last iterator of range.
    //----------------------------------------
    template<typename InputIt>
    SharedBuffer(InputIt first, InputIt last) : SharedBuffer{BufferType{first, last}}
    {}

    //========================================
    //! \brief Wrapping buffer initializer_list constructor.
    //! \param[in] init  Binary std::initializer_list of char.
    //----------------------------------------
    SharedBuffer(std::initializer_list<char> init);

    //========================================
    //! \brief Move constructor.
    //! \param[in] sharedBuffer  SharedBuffer to move.
    //----------------------------------------
    SharedBuffer(SharedBuffer&& sharedBuffer);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] sharedBuffer  SharedBuffer to copy.
    //----------------------------------------
    SharedBuffer(const SharedBuffer& sharedBuffer);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~SharedBuffer() = default;

public:
    //========================================
    //! \brief Returns a reference to the element at specified location pos. No bounds checking is performed.
    //! \param[in] index  Position of the element to return
    //! \return Reference to the requested element.
    //----------------------------------------
    BufferType::reference& operator[](const BufferType::size_type index);

    //========================================
    //! \brief Returns a reference to the element at specified location pos. No bounds checking is performed.
    //! \param[in] index  Position of the element to return
    //! \return Reference to the requested element.
    //----------------------------------------
    BufferType::const_reference& operator[](const BufferType::size_type index) const;

    //========================================
    //! \brief Move assigment operator.
    //! \param[in] sharedBuffer Buffer to move.
    //! \return Reference of this.
    //----------------------------------------
    SharedBuffer& operator=(SharedBuffer&& sharedBuffer);

    //========================================
    //! \brief Copy assigment operator.
    //! \param[in] sharedBuffer Buffer to copy.
    //! \return Reference of this.
    //----------------------------------------
    SharedBuffer& operator=(const SharedBuffer& sharedBuffer);

    //========================================
    //! \brief Checks if the shared buffer is empty.
    //! \return Either \c true if is empty or otherwise \c false.
    //----------------------------------------
    bool operator!() const;

    //========================================
    //! \brief Checks if the shared buffer is not empty.
    //! \return Either \c true if is not empty or otherwise \c false.
    //----------------------------------------
    explicit operator bool() const;

    //========================================
    //! \brief Equal compare of selected buffer section.
    //! \param[in] lhs  Buffer to compare.
    //! \param[in] rhs  Buffer to compare.
    //! \note Offset wont compare because of section compare.
    //! \return Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const SharedBuffer& lhs, const SharedBuffer& rhs);

    //========================================
    //! \brief Unequal compare of selected buffer section.
    //! \param[in] lhs  Buffer to compare.
    //! \param[in] rhs  Buffer to compare.
    //! \note Offset wont compare because of section compare.
    //! \return Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const SharedBuffer& lhs, const SharedBuffer& rhs);

public:
    //========================================
    //! \brief Select a section of the buffer with the offset and optional size.
    //! \param[in] offset   Absolute offset where the new section starts.
    //! \param[in] size     (Optional) Size of the section. Default: 0.
    //! \note Offset and size are absolute to the whole size of previous existing
    //! (possibly already offset) buffer.
    //! \note If the size is 0 the whole size minus offset will used.
    //! \return The sharedBuffer with the new selected section.
    //----------------------------------------
    SharedBuffer select(const std::size_t offset, const std::size_t size = 0) const;

public: // std vector interface
    //========================================
    //! \brief Get the first iterator of the selected section.
    //! \return Iterator of wrapped buffer.
    //----------------------------------------
    BufferType::iterator begin();

    //========================================
    //! \brief Get the first readonly iterator of the selected section.
    //! \return Readonly iterator of wrapped buffer.
    //----------------------------------------
    const BufferType::const_iterator cbegin() const;

    //========================================
    //! \brief Get the last iterator of the selected section.
    //! \note The selection ends before this iterator.
    //! \return Iterator of wrapped buffer.
    //----------------------------------------
    BufferType::iterator end();

    //========================================
    //! \brief Get the last readonly iterator of the selected section.
    //! \note The selection ends before this iterator.
    //! \return Readonly iterator of wrapped buffer.
    //----------------------------------------
    const BufferType::const_iterator cend() const;

    //========================================
    //! \brief Get pointer the first char of the selected section.
    //! \return Char pointer.
    //----------------------------------------
    char* data();

    //========================================
    //! \brief Get pointer the first char of the selected section.
    //! \return Readonly Char pointer.
    //----------------------------------------
    const char* data() const;

    //========================================
    //! \brief Get offset of the selected section.
    //! \return Offset where the selected section starts.
    //----------------------------------------
    std::size_t offset() const;

    //========================================
    //! \brief Checks wheter the selected section is of size 0.
    //! \return Either \c true if selected section is zero size or otherwise \c false.
    //----------------------------------------
    bool empty() const;

    //========================================
    //! \brief Get size of the selected section.
    //! \return Size of selected section.
    //! \attention Does not return the size of the underlying buffer.
    //----------------------------------------
    std::size_t size() const;

    //========================================
    //! \brief Resize the buffer and change the section if out of range.
    //! \params[in] newSize  New size of whole buffer.
    //! \note Does not behave like a standard container resize().
    //! \attention Resizing does not have an effect on the selection but on the underlying buffer (which may not even
    //! belong to you - so use carefully!). Select the new payload and size with select().
    //! \attention Method size() subtracts offset of the internal buffer, resize() does not! So the sizes do not match if the buffer has an internal offset!
    //----------------------------------------
    void resize(const std::size_t newSize);

public: // std shared_pointer interface
    //========================================
    //! \brief Get pointer to wrapped buffer.
    //! \return Buffer pointer.
    //----------------------------------------
    BufferType* get() const noexcept;

    //========================================
    //! \brief Get reference to wrapped buffer.
    //! \return Buffer reference.
    //----------------------------------------
    BufferType& operator*() const noexcept;

    //========================================
    //! \brief Get pointer to wrapped buffer.
    //! \return Buffer pointer.
    //----------------------------------------
    BufferType* operator->() const noexcept;

    //========================================
    //! \brief Reset buffer and selection.
    //! \param[in] newBuffer  Buffer pointer, if nullptr the buffer will constructed empty.
    //----------------------------------------
    void reset(BufferType* newBuffer = nullptr);

private:
    //========================================
    //! \brief Wrapped buffer pointer with shared ownership.
    //----------------------------------------
    BufferPointerType m_buffer;

    //========================================
    //! \brief Offset of the selected section.
    //----------------------------------------
    std::size_t m_offset;

    //========================================
    //! \brief Size of the selected section.
    //! \note If zero the size will be the size of the buffer minus offset.
    //----------------------------------------
    std::size_t m_size;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
