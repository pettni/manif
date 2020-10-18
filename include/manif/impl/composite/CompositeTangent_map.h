#ifndef _MANIF_MANIF_COMPOSITETANGENT_MAP_H_
#define _MANIF_MANIF_COMPOSITETANGENT_MAP_H_

#include "manif/impl/composite/CompositeTangent.h"

namespace manif
{
namespace internal
{

//! @brief traits specialization for Eigen Map
template<typename _Scalar, template<typename> class ... T>
struct traits<Eigen::Map<CompositeTangent<_Scalar, T...>, 0>>
  : public traits<CompositeTangent<_Scalar, T...>>
{
  using typename traits<CompositeTangent<_Scalar, T...>>::Scalar;
  using traits<CompositeTangent<Scalar, T...>>::DoF;
  using Base = CompositeTangentBase<Eigen::Map<CompositeTangent<Scalar, T...>, 0>>;
  using DataType = Eigen::Map<Eigen::Matrix<Scalar, DoF, 1>, 0>;
};

//! @brief traits specialization for Eigen Map const
template<typename _Scalar, template<typename> class ... T>
struct traits<Eigen::Map<const CompositeTangent<_Scalar, T...>, 0>>
  : public traits<const CompositeTangent<_Scalar, T...>>
{
  using typename traits<const CompositeTangent<_Scalar, T...>>::Scalar;
  using traits<const CompositeTangent<Scalar, T...>>::DoF;
  using Base = CompositeTangentBase<Eigen::Map<const CompositeTangent<Scalar, T...>, 0>>;
  using DataType = Eigen::Map<const Eigen::Matrix<Scalar, DoF, 1>, 0>;
};

}  // namespace internal
}  // namespace manif


namespace Eigen
{

/**
 * @brief Specialization of Map for manif::Composite
 */
template<class _Scalar, template<typename> class ... T>
class Map<manif::CompositeTangent<_Scalar, T...>, 0>
  : public manif::CompositeTangentBase<Map<manif::CompositeTangent<_Scalar, T...>, 0>>
{
  using Base = manif::CompositeTangentBase<Map<manif::CompositeTangent<_Scalar, T...>, 0>>;

public:
  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_API
  MANIF_INHERIT_TANGENT_OPERATOR

  Map(Scalar * coeffs)
  : data_(coeffs) {}

  DataType & coeffs() {return data_;}

  const DataType & coeffs() const {return data_;}

protected:
  DataType data_;
};

/**
 * @brief Specialization of Map for const manif::CompositeTangent
 */
template<class _Scalar, template<typename> class ... T>
class Map<const manif::CompositeTangent<_Scalar, T...>, 0>
  : public manif::CompositeTangentBase<Map<const manif::CompositeTangent<_Scalar, T...>, 0>>
{
  using Base = manif::CompositeTangentBase<Map<const manif::CompositeTangent<_Scalar, T...>, 0>>;

public:
  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_API
  MANIF_INHERIT_TANGENT_OPERATOR

  Map(const Scalar * coeffs)
  : data_(coeffs) {}

  const DataType & coeffs() const {return data_;}

protected:
  const DataType data_;
};

}  // namespace Eigen

#endif  // _MANIF_MANIF_COMPOSITETANGENT_MAP_H_
