#ifndef _MANIF_MANIF_COMPOSITE_PROPERTIES_H_
#define _MANIF_MANIF_COMPOSITE_PROPERTIES_H_

#include "manif/impl/traits.h"

namespace manif
{

// Forward declaration for type traits specialization
template<typename _Derived>
struct CompositeBase;
template<typename _Derived>
struct CompositeTangentBase;

template<typename _Scalar, template<typename> class ... _T>
struct Composite;
template<typename _Scalar, template<typename> class ... _T>
struct CompositeTangent;


namespace internal
{
namespace composite
{

// std::integer_sequence-equivalent
template<typename _Int, _Int ... _I>
struct intseq
{
  using value_type = _Int;
  static constexpr std::size_t size() noexcept {return sizeof...(_I);}
};

// join two intseqs
template<typename _Seq1, typename _Seq2>
struct intseq_join;

template<typename _Int, template<typename, _Int ...> class _IntSeq, _Int ... _I1, _Int ... _I2>
struct intseq_join<_IntSeq<_Int, _I1...>, _IntSeq<_Int, _I2...>>
{
  using type = _IntSeq<_Int, _I1..., _I2...>;
};

template<typename _Seq1, typename _Seq2>
using intseq_join_t = typename intseq_join<_Seq1, _Seq2>::type;


// create intseq of given length
template<typename _Int, size_t _N>
struct make_intseq;

template<typename _Int>
struct make_intseq<_Int, 0>
{
  using type = intseq<_Int>;
};

template<typename _Int, size_t _N>
struct make_intseq
{
  using type =
    typename intseq_join<typename make_intseq<_Int, _N - 1>::type, intseq<_Int, _N - 1>>::type;
};

template<typename _Int, size_t _N>
using make_intseq_t = typename make_intseq<_Int, _N>::type;


// extract element from integer sequence
template<size_t _Idx, typename _Seq>
struct intseq_element;

template<typename _Int, template<typename, _Int ...> class _IntSeq, _Int _I, _Int ... _Is>
struct intseq_element<0, _IntSeq<_Int, _I, _Is...>>
{
  static constexpr _Int value = _I;
};

template<
  typename _Int,
  template<typename, _Int ...> class _IntSeq,
  _Int _I,
  _Int ... _Is,
  size_t _Idx>
struct intseq_element<_Idx, _IntSeq<_Int, _I, _Is...>>
{
  static constexpr _Int value = intseq_element<_Idx - 1, _IntSeq<_Int, _Is...>>::value;
};

// c++14 only
// template<size_t _Idx, typename _Seq>
// constexpr typename _Seq::value_type intseq_element_v = intseq_element<_Idx, _Seq>::value;


// sum an integer sequence
template<typename _Seq>
struct intseq_sum;

template<typename _Int, template<typename, _Int ...> class _IntSeq>
struct intseq_sum<_IntSeq<_Int>>
{
  static constexpr _Int value = 0;
};

template<typename _Int, template<typename, _Int ...> class _IntSeq, _Int I, _Int ... Is>
struct intseq_sum<_IntSeq<_Int, I, Is...>>
{
  static constexpr _Int value = I + intseq_sum<_IntSeq<_Int, Is...>>::value;
};

// c++14 only
// template<typename _Seq>
// constexpr typename _Seq::value_type intseq_sum_v = intseq_sum<_Seq>::value;


// prefix-sum an integer sequence
template<typename _Int, typename _Collected, typename _Remaining, _Int Sum>
struct intseq_psum_impl;

template<
  typename _Int,
  template<typename, _Int ...> class _IntSeq,
  _Int... _Cur,
  _Int _Sum>
struct intseq_psum_impl<_Int, _IntSeq<_Int, _Cur...>, _IntSeq<_Int>, _Sum>
{
  using type = _IntSeq<_Int, _Cur...>;
};

template<typename _Int, template<typename, _Int ...> class _IntSeq,
  _Int... _Cur, _Int _First, _Int... _Rem, _Int _Sum>
struct intseq_psum_impl<_Int, _IntSeq<_Int, _Cur...>, _IntSeq<_Int, _First, _Rem...>, _Sum>
  : intseq_psum_impl<_Int, _IntSeq<_Int, _Cur..., _Sum>, _IntSeq<_Int, _Rem...>, _Sum + _First>
{};

template<class _Seq>
struct intseq_psum;

template<typename _Int, template<typename, _Int ...> class _IntSeq, _Int ... _I>
struct intseq_psum<_IntSeq<_Int, _I...>>
{
  using type = typename intseq_psum_impl<_Int, _IntSeq<_Int>, _IntSeq<_Int, _I...>, 0>::type;
};

template<class _Seq>
using intseq_psum_t = typename intseq_psum<_Seq>::type;

}  // namespace composite
}  // namespace internal


// SELECT INTEGER SEQUENCE TYPE
template<int ... _I>
using m_intseq = internal::composite::intseq<int, _I...>;

// c++14 only
// using m_intseq = std::integer_sequence<_Int, _I...>;

template<std::size_t _N>
using m_make_intseq = internal::composite::make_intseq_t<int, _N>;

// c++14 only
// using m_make_intseq = std::make_integer_sequence<_Int, _N>;

}  // namespace manif

#endif  // _MANIF_MANIF_COMPOSITE_PROPERTIES_H_
