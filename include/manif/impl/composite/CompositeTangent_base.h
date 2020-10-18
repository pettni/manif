#ifndef _MANIF_MANIF_COMPOSITETANGENT_BASE_H_
#define _MANIF_MANIF_COMPOSITETANGENT_BASE_H_

#include "manif/impl/composite/Composite_properties.h"
#include "manif/impl/tangent_base.h"

#include <utility>

namespace manif
{

/**
 * @brief The base class of the composite tangent.
 */
template<typename _Derived>
struct CompositeTangentBase : TangentBase<_Derived>
{
private:
  using Base = TangentBase<_Derived>;
  using Type = CompositeTangentBase<_Derived>;

  using IdxList = typename internal::traits<_Derived>::IdxList;

  using BegDoF = typename internal::traits<_Derived>::BegDoF;
  using LenDoF = typename internal::traits<_Derived>::LenDoF;

  using BegRep = typename internal::traits<_Derived>::BegRep;
  using LenRep = typename internal::traits<_Derived>::LenRep;

  using BegAlg = typename internal::traits<_Derived>::BegAlg;
  using LenAlg = typename internal::traits<_Derived>::LenAlg;

  template<int _Idx>
  using PartType = typename internal::traits<_Derived>::template PartType<_Idx>;

public:
  MANIF_TANGENT_TYPEDEF;
  MANIF_INHERIT_TANGENT_OPERATOR;

  using Base::coeffs;

  CompositeTangentBase()  = default;
  ~CompositeTangentBase() = default;


  // Tangent common API

  /**
   * @brief Hat operator.
   * @return An element of the Lie algebra.
   */
  LieAlg hat() const;

  /**
   * @brief Exponential operator.
   */
  LieGroup exp(OptJacobianRef J_m_t = {}) const;

  /**
   * @brief This function is deprecated.
   * Please considere using
   * @ref exp instead.
   */
  MANIF_DEPRECATED
  LieGroup retract(OptJacobianRef J_m_t = {}) const;

  /**
   * @brief Get the right Jacobian.
   */
  Jacobian rjac() const;

  /**
   * @brief Get the left Jacobian.
   */
  Jacobian ljac() const;

  /**
   * @brief Get the inverse of the right Jacobian.
   */
  Jacobian rjacinv() const;

  /**
   * @brief Get the inverse of the right Jacobian.
   */
  Jacobian ljacinv() const;

  /**
   * @brief
   */
  Jacobian smallAdj() const;


  // CompositeTangent specific API

  //! Get a map to a part of the composite tangent
  template<int _Idx>
  Eigen::Map<PartType<_Idx>> get();

  //! Get a const map to a part of the composite tangent
  template<int _Idx>
  Eigen::Map<const PartType<_Idx>> get() const;

protected:
  template<int ... _Idx, int ... _BegAlg, int ... _LenAlg>
  LieAlg
  hat_impl(m_intseq<_Idx...>, m_intseq<_BegAlg...>, m_intseq<_LenAlg...>) const;

  template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
  LieGroup exp_impl(
    OptJacobianRef J_m_t, m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const;

  template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
  Jacobian
  rjac_impl(m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const;

  template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
  Jacobian
  ljac_impl(m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const;

  template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
  Jacobian
  rjacinv_impl(m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const;

  template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
  Jacobian
  ljacinv_impl(m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const;

  template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
  Jacobian
  smallAdj_impl(m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const;

  friend internal::GeneratorEvaluator<CompositeTangentBase<_Derived>>;
  friend internal::RandomEvaluatorImpl<CompositeTangentBase<_Derived>>;
};


// Tangent common API

template<typename _Derived>
typename CompositeTangentBase<_Derived>::LieAlg
CompositeTangentBase<_Derived>::hat() const
{
  return hat_impl(IdxList{}, BegAlg{}, LenAlg{});
}

template<typename _Derived>
template<int ... _Idx, int ... _BegAlg, int ... _LenAlg>
typename CompositeTangentBase<_Derived>::LieAlg
CompositeTangentBase<_Derived>::hat_impl(
  m_intseq<_Idx...>, m_intseq<_BegAlg...>, m_intseq<_LenAlg...>) const
{
  LieAlg ret = LieAlg::Zero();
  // c++11 "fold expression"
  auto l = {((ret.template block<_LenAlg, _LenAlg>(_BegAlg, _BegAlg) = get<_Idx>().hat()), 0) ...};
  static_cast<void>(l);  // compiler warning
  return ret;
}

template<typename _Derived>
typename CompositeTangentBase<_Derived>::LieGroup
CompositeTangentBase<_Derived>::exp(OptJacobianRef J_m_t) const
{
  if (J_m_t) {
    J_m_t->setZero();
  }
  return exp_impl(J_m_t, IdxList{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
typename CompositeTangentBase<_Derived>::LieGroup
CompositeTangentBase<_Derived>::exp_impl(
  OptJacobianRef J_m_t, m_intseq<_Idx...>,
  m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const
{
  if (J_m_t) {
    return LieGroup(get<_Idx>().exp(J_m_t->template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF)) ...);
  }
  return LieGroup(get<_Idx>().exp() ...);
}

template<typename _Derived>
typename CompositeTangentBase<_Derived>::LieGroup
CompositeTangentBase<_Derived>::retract(OptJacobianRef J_m_t) const
{
  return exp(J_m_t);
}

template<typename _Derived>
typename CompositeTangentBase<_Derived>::Jacobian
CompositeTangentBase<_Derived>::rjac() const
{
  return rjac_impl(IdxList{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
typename CompositeTangentBase<_Derived>::Jacobian
CompositeTangentBase<_Derived>::ljac() const
{
  return ljac_impl(IdxList{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
typename CompositeTangentBase<_Derived>::Jacobian
CompositeTangentBase<_Derived>::rjacinv() const
{
  return rjacinv_impl(IdxList{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
typename CompositeTangentBase<_Derived>::Jacobian
CompositeTangentBase<_Derived>::ljacinv() const
{
  return ljacinv_impl(IdxList{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
typename CompositeTangentBase<_Derived>::Jacobian
CompositeTangentBase<_Derived>::smallAdj() const
{
  return smallAdj_impl(IdxList{}, BegDoF{}, LenDoF{});
}

template<typename _Derived>
template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
typename CompositeTangentBase<_Derived>::Jacobian
CompositeTangentBase<_Derived>::rjac_impl(
  m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const
{
  Jacobian Jr = Jacobian::Zero();
  // c++11 "fold expression"
  auto l = {((Jr.template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF) = get<_Idx>().rjac() ), 0) ...};
  static_cast<void>(l);  // compiler warning
  return Jr;
}

template<typename _Derived>
template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
typename CompositeTangentBase<_Derived>::Jacobian
CompositeTangentBase<_Derived>::ljac_impl(
  m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const
{
  Jacobian Jr = Jacobian::Zero();
  // c++11 "fold expression"
  auto l = {((Jr.template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF) = get<_Idx>().ljac()), 0) ...};
  static_cast<void>(l);  // compiler warning
  return Jr;
}

template<typename _Derived>
template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
typename CompositeTangentBase<_Derived>::Jacobian
CompositeTangentBase<_Derived>::rjacinv_impl(
  m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const
{
  Jacobian Jr = Jacobian::Zero();
  // c++11 "fold expression"
  auto l = {
    ((Jr.template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF) = get<_Idx>().rjacinv()), 0) ...
  };
  static_cast<void>(l);  // compiler warning
  return Jr;
}

template<typename _Derived>
template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
typename CompositeTangentBase<_Derived>::Jacobian
CompositeTangentBase<_Derived>::ljacinv_impl(
  m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const
{
  Jacobian Jr = Jacobian::Zero();
  // c++11 "fold expression"
  auto l = {
    ((Jr.template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF) = get<_Idx>().ljacinv()), 0) ...
  };
  static_cast<void>(l);  // compiler warning
  return Jr;
}

template<typename _Derived>
template<int ... _Idx, int ... _BegDoF, int ... _LenDoF>
typename CompositeTangentBase<_Derived>::Jacobian
CompositeTangentBase<_Derived>::smallAdj_impl(
  m_intseq<_Idx...>, m_intseq<_BegDoF...>, m_intseq<_LenDoF...>) const
{
  Jacobian Jr = Jacobian::Zero();
  // c++11 "fold expression"
  auto l = {
    ((Jr.template block<_LenDoF, _LenDoF>(_BegDoF, _BegDoF) = get<_Idx>().smallAdj()), 0) ...
  };
  static_cast<void>(l);  // compiler warning
  return Jr;
}


// CompositeTangent specific API

template<typename _Derived>
template<int _Idx>
Eigen::Map<typename CompositeTangentBase<_Derived>::template PartType<_Idx>>
CompositeTangentBase<_Derived>::get()
{
  return Eigen::Map<PartType<_Idx>>(
    static_cast<_Derived &>(*this).coeffs().data() +
    internal::composite::intseq_element<_Idx, BegRep>::value);
}

template<typename _Derived>
template<int _Idx>
Eigen::Map<const typename CompositeTangentBase<_Derived>::template PartType<_Idx>>
CompositeTangentBase<_Derived>::get() const
{
  return Eigen::Map<const PartType<_Idx>>(
    static_cast<const _Derived &>(*this).coeffs().data() +
    internal::composite::intseq_element<_Idx, BegRep>::value);
}

namespace internal
{

/**
 * @brief Generator specialization for CompositeTangentBase objects.
 */
template<typename Derived>
struct GeneratorEvaluator<CompositeTangentBase<Derived>>
{
  static typename CompositeTangentBase<Derived>::LieAlg
  run(const unsigned int i)
  {
    MANIF_CHECK(
      i < CompositeTangentBase<Derived>::DoF,
      "Index i must less than DoF!",
      invalid_argument);

    return run(
      i,
      typename CompositeTangentBase<Derived>::IdxList{},
      typename CompositeTangentBase<Derived>::BegDoF{},
      typename CompositeTangentBase<Derived>::LenDoF{},
      typename CompositeTangentBase<Derived>::BegAlg{},
      typename CompositeTangentBase<Derived>::LenAlg{});
  }

  template<int ... _Idx, int ... _BegDoF, int ... _LenDoF, int ... _BegAlg, int ... _LenAlg>
  static typename CompositeTangentBase<Derived>::LieAlg
  run(
    const unsigned int i, m_intseq<_Idx...>,
    m_intseq<_BegDoF...>, m_intseq<_LenDoF...>,
    m_intseq<_BegAlg...>, m_intseq<_LenAlg...>)
  {
    using LieAlg = typename CompositeTangentBase<Derived>::LieAlg;
    LieAlg Ei = LieAlg::Constant(0);
    // c++11 "fold expression"
    auto l = {((Ei.template block<_LenAlg, _LenAlg>(_BegAlg, _BegAlg) =
      (i >= _BegDoF && i < _BegDoF + _LenDoF) ?
      CompositeTangentBase<Derived>::template PartType<_Idx>::Generator(i - _BegDoF) :
      CompositeTangentBase<Derived>::template PartType<_Idx>::LieAlg::Zero()
      ), 0) ...};
    static_cast<void>(l);  // compiler warning
    return Ei;
  }
};

//! @brief Random specialization for CompositeTangentBase objects.
template<typename Derived>
struct RandomEvaluatorImpl<CompositeTangentBase<Derived>>
{
  static void run(CompositeTangentBase<Derived> & m)
  {
    run(m, typename CompositeTangentBase<Derived>::IdxList{});
  }

  template<int ... _Idx>
  static void run(CompositeTangentBase<Derived> & m, m_intseq<_Idx...>)
  {
    m = typename CompositeTangentBase<Derived>::Tangent(
      CompositeTangentBase<Derived>::template PartType<_Idx>::Random() ...);
  }
};

}  // namespace internal
}  // namespace manif

#endif  // _MANIF_MANIF_COMPOSITETANGENT_BASE_H_
