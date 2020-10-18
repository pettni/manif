#ifndef _MANIF_MANIF_COMPOSITE_MAP_H_
#define _MANIF_MANIF_COMPOSITE_MAP_H_

#include "manif/impl/composite/Composite.h"

namespace manif
{
namespace internal
{

//! @brief traits specialization for Eigen Map
template<typename _Scalar, template<typename> class ... T>
struct traits<Eigen::Map<Composite<_Scalar, T...>, 0>>
  : public traits<Composite<_Scalar, T...>>
{
  using typename traits<Composite<_Scalar, T...>>::Scalar;
  using traits<Composite<Scalar, T...>>::RepSize;
  using Base = CompositeBase<Eigen::Map<Composite<Scalar, T...>, 0>>;
  using DataType = Eigen::Map<Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

//! @brief traits specialization for Eigen Map const
template<typename _Scalar, template<typename> class ... T>
struct traits<Eigen::Map<const Composite<_Scalar, T...>, 0>>
  : public traits<const Composite<_Scalar, T...>>
{
  using typename traits<const Composite<_Scalar, T...>>::Scalar;
  using traits<const Composite<Scalar, T...>>::RepSize;
  using Base = CompositeBase<Eigen::Map<const Composite<Scalar, T...>, 0>>;
  using DataType = Eigen::Map<const Eigen::Matrix<Scalar, RepSize, 1>, 0>;
};

}  // namespace internal
}  // namespace manif


namespace Eigen
{

/**
 * @brief Specialization of Map for manif::Composite
 */
template<class _Scalar, template<typename> class ... T>
class Map<manif::Composite<_Scalar, T...>, 0>
  : public manif::CompositeBase<Map<manif::Composite<_Scalar, T...>, 0>>
{
  using Base = manif::CompositeBase<Map<manif::Composite<_Scalar, T...>, 0>>;

public:
  MANIF_COMPLETE_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_API

  Map(Scalar * coeffs)
  : data_(coeffs) {}

  DataType & coeffs() {return data_;}

  const DataType & coeffs() const {return data_;}

protected:
  DataType data_;
};

/**
 * @brief Specialization of Map for const manif::Composite
 */
template<class _Scalar, template<typename> class ... T>
class Map<const manif::Composite<_Scalar, T...>, 0>
  : public manif::CompositeBase<Map<const manif::Composite<_Scalar, T...>, 0>>
{
  using Base = manif::CompositeBase<Map<const manif::Composite<_Scalar, T...>, 0>>;

public:
  MANIF_COMPLETE_GROUP_TYPEDEF
  MANIF_INHERIT_GROUP_API

  Map(const Scalar * coeffs)
  : data_(coeffs) {}

  const DataType & coeffs() const {return data_;}

protected:
  const DataType data_;
};

}  // namespace Eigen

#endif  // _MANIF_MANIF_COMPOSITE_MAP_H_
