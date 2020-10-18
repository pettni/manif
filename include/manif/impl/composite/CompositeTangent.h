#ifndef _MANIF_MANIF_COMPOSITETANGENT_H_
#define _MANIF_MANIF_COMPOSITETANGENT_H_

#include "manif/impl/composite/CompositeTangent_base.h"

#include <tuple>
#include <utility>

namespace manif
{
namespace internal
{

//! Traits specialization
template<typename _Scalar, template<typename> class ... _T>
struct traits<CompositeTangent<_Scalar, _T...>>
{
  // Composite-specific traits
  using IdxList = m_make_intseq<sizeof...(_T)>;

  using LenDim = m_intseq<_T<_Scalar>::Tangent::Dim ...>;
  using BegDim = composite::intseq_psum_t<LenDim>;

  using LenDoF = m_intseq<_T<_Scalar>::Tangent::DoF ...>;
  using BegDoF = composite::intseq_psum_t<LenDoF>;

  using LenRep = m_intseq<_T<_Scalar>::Tangent::RepSize ...>;
  using BegRep = composite::intseq_psum_t<LenRep>;

  using LenAlg = m_intseq<_T<_Scalar>::Tangent::LieAlg::RowsAtCompileTime ...>;
  using BegAlg = composite::intseq_psum_t<LenAlg>;

  template<size_t Idx>
  using PartType = typename std::tuple_element<
    Idx, std::tuple<typename _T<_Scalar>::Tangent ...>
    >::type;

  // Regular traits
  using Scalar = _Scalar;

  using LieGroup = Composite<_Scalar, _T...>;
  using Tangent = CompositeTangent<_Scalar, _T...>;

  using Base = CompositeTangentBase<Tangent>;

  static constexpr int Dim = composite::intseq_sum<LenDim>::value;
  static constexpr int DoF = composite::intseq_sum<LenDoF>::value;
  static constexpr int RepSize = composite::intseq_sum<LenRep>::value;

  using DataType = Eigen::Matrix<Scalar, RepSize, 1>;
  using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
  using LieAlg = Eigen::Matrix<Scalar, composite::intseq_sum<LenAlg>::value,
      composite::intseq_sum<LenAlg>::value>;
};

}  // namespace internal

//
// CompositeTangent
//

/**
 * @brief Represents an element of the tangent space
 */
template<typename _Scalar, template<typename> class ... _T>
struct CompositeTangent : CompositeTangentBase<CompositeTangent<_Scalar, _T...>>
{
private:
  static_assert(sizeof...(_T) > 0, "Must have at least one element !");

  using Base = CompositeTangentBase<CompositeTangent<_Scalar, _T...>>;
  using Type = CompositeTangent<_Scalar, _T...>;

  using BegRep = typename internal::traits<CompositeTangent>::BegRep;
  using LenRep = typename internal::traits<CompositeTangent>::LenRep;

public:
  MANIF_MAKE_ALIGNED_OPERATOR_NEW_COND

  MANIF_TANGENT_TYPEDEF
  MANIF_INHERIT_TANGENT_API
  MANIF_INHERIT_TANGENT_OPERATOR

  CompositeTangent()  = default;
  ~CompositeTangent() = default;

  // Copy constructor given base
  CompositeTangent(const Base & o);

  template<typename _DerivedOther>
  CompositeTangent(const CompositeTangentBase<_DerivedOther> & o);

  template<typename _DerivedOther>
  CompositeTangent(const TangentBase<_DerivedOther> & o);

  // Copy constructor given Eigen
  template<typename _EigenDerived>
  CompositeTangent(const Eigen::MatrixBase<_EigenDerived> & theta);


  // Tangent common API

  DataType & coeffs();

  const DataType & coeffs() const;


  // CompositeTangent specific API

  //! Construct from individual parts
  CompositeTangent(typename _T<_Scalar>::Tangent && ... parts);

protected:
  // Helper for the parts constructor
  template<int ... _BegRep, int ... _LenRep>
  CompositeTangent(
    m_intseq<_BegRep...>, m_intseq<_LenRep...>, typename _T<_Scalar>::Tangent && ... parts);

protected:
  DataType data_;
};

MANIF_EXTRA_GROUP_TYPEDEF(CompositeTangent)

template<typename _Scalar, template<typename> class ... _T>
CompositeTangent<_Scalar, _T...>::CompositeTangent(const Base & o)
: data_(o.coeffs())
{}

template<typename _Scalar, template<typename> class ... _T>
template<typename _DerivedOther>
CompositeTangent<_Scalar, _T...>::CompositeTangent(const CompositeTangentBase<_DerivedOther> & o)
: data_(o.coeffs())
{}

template<typename _Scalar, template<typename> class ... _T>
template<typename _DerivedOther>
CompositeTangent<_Scalar, _T...>::CompositeTangent(const TangentBase<_DerivedOther> & o)
: data_(o.coeffs())
{}

template<typename _Scalar, template<typename> class ... _T>
template<typename _EigenDerived>
CompositeTangent<_Scalar, _T...>::CompositeTangent(const Eigen::MatrixBase<_EigenDerived> & data)
: data_(data)
{}

template<typename _Scalar, template<typename> class ... _T>
CompositeTangent<_Scalar, _T...>::CompositeTangent(typename _T<_Scalar>::Tangent && ... parts)
: CompositeTangent(BegRep{}, LenRep{}, std::forward<typename _T<_Scalar>::Tangent>(parts) ...)
{}

template<typename _Scalar, template<typename> class ... _T>
template<int ... _BegRep, int ... _LenRep>
CompositeTangent<_Scalar, _T...>::CompositeTangent(
  m_intseq<_BegRep...>, m_intseq<_LenRep...>,
  typename _T<_Scalar>::Tangent && ... parts)
{
  // c++11 "fold expression"
  auto l = {((data_.template segment<_LenRep>(_BegRep) =
    std::forward<typename _T<_Scalar>::Tangent>(parts).coeffs()), 0) ...};
  static_cast<void>(l);  // compiler warning
}

template<typename _Scalar, template<typename> class ... _T>
typename CompositeTangent<_Scalar, _T...>::DataType &
CompositeTangent<_Scalar, _T...>::coeffs()
{
  return data_;
}

template<typename _Scalar, template<typename> class ... _T>
const typename CompositeTangent<_Scalar, _T...>::DataType &
CompositeTangent<_Scalar, _T...>::coeffs() const
{
  return data_;
}

}  // namespace manif

#endif  // _MANIF_MANIF_COMPOSITETANGENT_H_
