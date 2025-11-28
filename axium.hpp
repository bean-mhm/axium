// axium 0.3.0
// https://github.com/bean-mhm/axium
//
// axium (stylized as lowercase) is a single-header C++ math library providing
// utility classes and functions for:
// - Generic math functions
// - Matrices
// - Vectors (2D, 3D, 4D)
// - Quaternions
// - Polar and spherical coordinates
// - 3D rays
// - Basic shapes and their intersections:
//   - Axis-Aligned Bounding Boxes (AABB)
//   - Circles
//   - Spheres
// - 2D and 3D linear transformations
// - Smart number to string conversion

// Zero-Clause BSD
// =============
// 
// Permission to use, copy, modify, and/or distribute this software for
// any purpose with or without fee is hereby granted.
// 
// THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
// WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
// FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
// DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
// AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
// OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <iterator>
#include <concepts>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace axium
{

    // MARK: primitive types
    using i8 = int8_t;
    using u8 = uint8_t;
    using i16 = int16_t;
    using u16 = uint16_t;
    using i32 = int32_t;
    using u32 = uint32_t;
    using i64 = int64_t;
    using u64 = uint64_t;
    using isize = ptrdiff_t;
    using usize = size_t;
    using f32 = float;
    using f64 = double;

    // MARK: constants
    template<std::floating_point T>
    static constexpr T INF = std::numeric_limits<T>::infinity();
    template<std::floating_point T>
    static constexpr T SQRT2 = 1.414213562373095048801688724;
    template<std::floating_point T>
    static constexpr T SQRT3 = 1.732050807568877293527446341;
    template<std::floating_point T>
    static constexpr T E = 2.718281828459045235360287471;
    template<std::floating_point T>
    static constexpr T PI = 3.141592653589793238462643383;
    template<std::floating_point T>
    static constexpr T TAU = 6.283185307179586476925286766;
    template<std::floating_point T>
    static constexpr T PI_OVER_2 = 1.570796326794896619231321691;
    template<std::floating_point T>
    static constexpr T DEG2RAD = 0.017453292519943295769236907;
    template<std::floating_point T>
    static constexpr T RAD2DEG = 57.29577951308232087679815481;

    // MARK: math functions

    // for use in to_string()
    template<std::floating_point T>
    i32 determine_precision_for_string(
        T v,
        i32 max_significant_digits = 4, // can only reduce decimal digits
        i32 min_precision = 1,
        i32 max_precision = 7
    )
    {
        // this may be negative and it's intentional
        i32 n_integral_digits = (i32)std::floor(std::log10(v)) + 1;

        min_precision = std::max(min_precision, 0);
        max_precision = std::max(max_precision, min_precision);

        return std::clamp(
            max_significant_digits - n_integral_digits,
            min_precision,
            max_precision
        );
    }

    template<std::integral T>
    std::string to_string(T v)
    {
        return std::to_string(v);
    }

    template<std::floating_point T>
    std::string to_string(
        T v,
        i32 max_significant_digits = 4, // can only reduce decimal digits
        i32 min_precision = 1,
        i32 max_precision = 7,
        i32* out_precision = nullptr,
        i32* out_n_trailing_zeros = nullptr
    )
    {
        i32 precision = determine_precision_for_string(
            v,
            max_significant_digits,
            min_precision,
            max_precision
        );

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision) << v;
        std::string s = oss.str();

        // remove redundant trailing zeros after decimal point
        i32 n_trailing_zeros = 0;
        if (precision > 0)
        {
            while (s.ends_with('0'))
            {
                s = s.substr(0, s.length() - 1);
                n_trailing_zeros++;
            }
            if (s.ends_with('.'))
            {
                s = s.substr(0, s.length() - 1);
            }
        }

        if (out_precision)
        {
            *out_precision = precision;
        }
        if (out_n_trailing_zeros)
        {
            *out_n_trailing_zeros = n_trailing_zeros;
        }

        if (s == "-0")
        {
            s = "0";
        }

        return s;
    }

    template<std::floating_point T>
    constexpr T radians(T degrees)
    {
        return degrees * DEG2RAD<T>;
    }

    template<std::floating_point T>
    constexpr T degrees(T radians)
    {
        return radians * RAD2DEG<T>;
    }

    template<std::floating_point T>
    inline T sin(T angle)
    {
        return std::sin(angle);
    }

    template<std::floating_point T>
    inline T cos(T angle)
    {
        return std::cos(angle);
    }

    template<std::floating_point T>
    inline T tan(T angle)
    {
        return std::tan(angle);
    }

    template<std::floating_point T>
    inline T asin(T x)
    {
        return std::asin(x);
    }

    template<std::floating_point T>
    inline T acos(T x)
    {
        return std::acos(x);
    }

    template<std::floating_point T>
    inline T atan(T y, T x)
    {
        return std::atan2(y, x);
    }

    template<std::floating_point T>
    inline T atan(T y_over_x)
    {
        return std::atan(y_over_x);
    }

    template<std::floating_point T>
    inline T sinh(T x)
    {
        return std::sinh(x);
    }

    template<std::floating_point T>
    inline T cosh(T x)
    {
        return std::cosh(x);
    }

    template<std::floating_point T>
    inline T tanh(T x)
    {
        return std::tanh(x);
    }

    template<std::floating_point T>
    inline T asinh(T x)
    {
        return std::asinh(x);
    }

    template<std::floating_point T>
    inline T acosh(T x)
    {
        return std::acosh(x);
    }

    template<std::floating_point T>
    inline T atanh(T x)
    {
        return std::atanh(x);
    }

    template<std::floating_point T>
    inline T pow(T x, T y)
    {
        return std::pow(x, y);
    }

    template<std::floating_point T>
    inline T exp(T x)
    {
        return std::exp(x);
    }

    template<std::floating_point T>
    inline T log(T x)
    {
        return std::log(x);
    }

    template<std::floating_point T>
    inline T exp2(T x)
    {
        return std::exp2(x);
    }

    template<std::floating_point T>
    inline T log2(T x)
    {
        return std::log2(x);
    }

    template<std::floating_point T>
    inline T squared(T x)
    {
        return x * x;
    }

    template<std::floating_point T>
    inline T sqrt(T x)
    {
        return std::sqrt(x);
    }

    template<std::floating_point T>
    inline T inversesqrt(T x)
    {
        return (T)1 / std::sqrt(x);
    }

    template<typename T>
    inline T abs(T x)
    {
        return std::abs(x);
    }

    template<typename T>
    constexpr T sign(T x)
    {
        if (x > 0) return 1;
        if (x == 0) return 0;
        if (x < 0) return -1;
    }

    template<std::floating_point T>
    inline T floor(T x)
    {
        return std::floor(x);
    }

    template<std::floating_point T>
    inline T ceil(T x)
    {
        return std::ceil(x);
    }

    template<std::floating_point T>
    inline T trunc(T x)
    {
        return std::trunc(x);
    }

    template<std::floating_point T>
    inline T fract(T x)
    {
        return x - std::floor(x);
    }

    template<std::floating_point T>
    inline T mod(T x, T y)
    {
        return std::fmod(x, y);
    }

    template<std::floating_point T>
    inline T modf(T x, T& i)
    {
        return std::modf(x, &i);
    }

    template<std::floating_point T>
    inline T wrap(T x, T start, T end)
    {
        return start + std::fmod(x - start, end - start);
    }

    template<std::integral T>
    inline T wrap(T x, T start, T end)
    {
        return start + ((x - start) % (end - start));
    }

    template<typename T>
    constexpr T min(T x, T y)
    {
        return std::min(x, y);
    }

    template<typename T>
    constexpr T max(T x, T y)
    {
        return std::max(x, y);
    }

    template<typename T>
    constexpr T clamp(T x, T min, T max)
    {
        return std::clamp(x, min, max);
    }

    template<std::floating_point T>
    constexpr T clamp01(T x)
    {
        return std::clamp(x, (T)0, (T)1);
    }

    template<std::floating_point T>
    constexpr T mix(T x, T y, T a)
    {
        return x + a * (y - x);
    }

    template<std::floating_point T>
    constexpr T remap(T x, T a_start, T a_end, T b_start, T b_end)
    {
        return
            b_start + ((b_end - b_start) / (a_end - a_start)) * (x - a_start);
    }

    template<std::floating_point T>
    constexpr T remap_clamp(
        T x,
        T a_start,
        T a_end,
        T b_start,
        T b_end
    )
    {
        T t = clamp01((x - a_start) / (a_end - a_start));
        return b_start + t * (b_end - b_start);
    }

    template<std::floating_point T>
    constexpr T remap01(T x, T a_start, T a_end)
    {
        return clamp01((x - a_start) / (a_end - a_start));
    }

    template<typename T>
    constexpr T step(T edge, T x)
    {
        if (x < edge) return 0;
        return 1;
    }

    template<std::floating_point T>
    constexpr T smoothstep(T edge0, T edge1, T x)
    {
        T t = clamp01((x - edge0) / (edge1 - edge0));
        return t * t * ((T)3 - (T)2 * t);
    }

    template<std::floating_point T>
    inline bool isnan(T x)
    {
        return std::isnan(x);
    }

    template<std::floating_point T>
    inline bool isinf(T x)
    {
        return std::isinf(x);
    }

    template<std::floating_point T>
    inline bool solve_quadratic(T a, T b, T c, T& t0, T& t1)
    {
        T discrim = b * b - (T)4. * a * c;
        if (discrim < 0)
            return false;
        T root_discrim = sqrt(discrim);
        T q;
        if (b < 0)
        {
            q = (T)(-.5) * (b - root_discrim);
        }
        else
        {
            q = (T)(-.5) * (b + root_discrim);
        }
        t0 = q / a;
        t1 = c / q;
        if (t0 > t1)
            std::swap(t0, t1);
        return true;
    }

    template<std::floating_point T>
    inline bool solve_linear_2x2(
        const T a[2][2],
        const T b[2],
        T& x0,
        T& x1
    )
    {
        T det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        if (abs(det) < (T)1e-10)
            return false;
        x0 = (a[1][1] * b[0] - a[0][1] * b[1]) / det;
        x1 = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
        if (isnan(x0) || isnan(x1))
            return false;
        return true;
    }

    // MARK: matrices

    // row-major matrix
    template<typename T, i32 n_row, i32 n_col>
    class Mat
    {
    public:
        constexpr Mat()
        {
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    m[row][col] = (row == col) ? (T)1 : (T)0;
                }
            }
        }

        constexpr Mat(const T mat[n_row][n_col])
        {
            std::copy(&mat[0][0], &mat[0][0] + (n_row * n_col), &m[0][0]);
        }

        constexpr Mat(const T* mat)
        {
            std::copy(mat, mat + (n_row * n_col), &m[0][0]);
        }

        constexpr Mat(const std::array<T, n_row* n_col>& mat)
        {
            std::copy(mat.data(), mat.data() + (n_row * n_col), &m[0][0]);
        }

        std::string to_string() const
        {
            std::string s("[");
            for (i32 row = 0; row < n_row; row++)
            {
                s += (row > 0 ? " [ " : "[ ");
                for (i32 col = 0; col < n_col; col++)
                {
                    s += to_string(m[row][col]);
                    if (col != n_col - 1)
                        s += "  ";
                }
                s += (row != n_row - 1 ? " ]\n" : " ]]");
            }
            return s;
        }

        friend std::ostream& operator<<(std::ostream& os, const Mat& m)
        {
            os << m.to_string();
            return os;
        }

        constexpr Mat operator*(T s) const
        {
            Mat r;
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    r(row, col) = (*this)(row, col) * s;
                }
            }
            return r;
        }

        constexpr Mat& operator*=(T s)
        {
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    (*this)(row, col) *= s;
                }
            }
            return *this;
        }

        constexpr Mat operator/(T s) const
        {
            Mat r;
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    r(row, col) = (*this)(row, col) / s;
                }
            }
            return r;
        }

        constexpr Mat& operator/=(T s)
        {
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    (*this)(row, col) /= s;
                }
            }
            return *this;
        }

        constexpr Mat operator+(const Mat& m) const
        {
            Mat r;
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    r(row, col) = (*this)(row, col) + m(row, col);
                }
            }
            return r;
        }

        constexpr Mat& operator+=(const Mat& m)
        {
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    (*this)(row, col) += m(row, col);
                }
            }
            return *this;
        }

        constexpr Mat operator-(const Mat& m) const
        {
            Mat r;
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    r(row, col) = (*this)(row, col) - m(row, col);
                }
            }
            return r;
        }

        constexpr Mat& operator-=(const Mat& m)
        {
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    (*this)(row, col) -= m(row, col);
                }
            }
            return *this;
        }

        template<i32 n2>
        constexpr Mat<T, n_row, n2> operator*(
            const Mat<T, n_col, n2>& m
            ) const
        {
            Mat<T, n_row, n2> r;
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n2; col++)
                {
                    T dot = 0;
                    for (i32 i = 0; i < n_col; i++)
                    {
                        dot += (*this)(row, i) * m(i, col);
                    }
                    r(row, col) = dot;
                }
            }
            return r;
        }

        constexpr bool operator==(const Mat& m2) const
        {
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    if (m[row][col] != m2.m[row][col])
                        return false;
                }
            }
            return true;
        }

        constexpr bool operator!=(const Mat& m2) const
        {
            for (i32 row = 0; row < n_row; row++)
            {
                for (i32 col = 0; col < n_col; col++)
                {
                    if (m[row][col] != m2.m[row][col])
                        return true;
                }
            }
            return false;
        }

        constexpr T operator()(i32 index) const
        {
            return (&m[0][0])[index];
        }

        constexpr T& operator()(i32 index)
        {
            return (&m[0][0])[index];
        }

        constexpr T operator()(i32 row, i32 col) const
        {
            return m[row][col];
        }

        constexpr T& operator()(i32 row, i32 col)
        {
            return m[row][col];
        }

        // sub-matrix
        // * the indices are inclusive. for example, if the start row and column
        //   indices are both 0, and the end row and column indices are both 2,
        //   this function will return the upper-left 3x3 portion.
        // * the value of end_row must not be smaller than the value of
        //   start_row, and the same goes for start_col and end_col.
        template<i32 start_row, i32 start_col, i32 end_row, i32 end_col>
        constexpr Mat<T, end_row - start_row + 1, end_col - start_col + 1>
            sub() const
        {
            Mat<T, end_row - start_row + 1, end_col - start_col + 1> r;
            for (i32 row = start_row; row <= end_row; row++)
            {
                for (i32 col = start_col; col <= end_col; col++)
                {
                    r(row - start_row, col - start_col) = (*this)(row, col);
                }
            }
        }

        // upper-left n x m sub-matrix
        // * n must be smaller than or equal to n_row.
        // * m must be smaller than or equal to n_col.
        template<i32 n, i32 m>
        constexpr Mat<T, n, m> sub() const
        {
            Mat<T, n, m> r;
            for (i32 row = 0; row < n; row++)
            {
                for (i32 col = 0; col < m; col++)
                {
                    r(row, col) = (*this)(row, col);
                }
            }
            return r;
        }

        static constexpr i32 n_rows()
        {
            return n_row;
        }

        static constexpr i32 n_cols()
        {
            return n_col;
        }

        static constexpr i32 n_elements()
        {
            return n_row * n_col;
        }

    private:
        T m[n_row][n_col];

    };

    template<typename T, i32 n_row, i32 n_col>
    constexpr Mat<T, n_row, n_col> operator*(
        T s,
        const Mat<T, n_row, n_col>& m
        )
    {
        return m * s;
    }

    template<typename T, i32 n>
    constexpr bool is_identity(const Mat<T, n, n>& m)
    {
        for (i32 row = 0; row < n; row++)
        {
            for (i32 col = 0; col < n; col++)
            {
                T expected = (row == col) ? (T)1 : (T)0;
                if (m(row, col) != expected)
                    return false;
            }
        }
        return true;
    }

    // cofactor of m[p][q]
    // https://www.geeksforgeeks.org/adjoint-inverse-matrix/
    template<typename T, i32 n>
    constexpr Mat<T, n - 1, n - 1> cofactor(
        const Mat<T, n, n>& m,
        i32 p,
        i32 q
    )
    {
        i32 i = 0, j = 0;
        Mat<T, n - 1, n - 1> r;
        for (i32 row = 0; row < n; row++)
        {
            for (i32 col = 0; col < n; col++)
            {
                // copy into the result matrix only those elements which are not
                // in the current row and column
                if (row != p && col != q)
                {
                    r(i, j++) = m(row, col);

                    // row is filled, so increase the row index and reset the
                    // col index
                    if (j == n - 1)
                    {
                        j = 0;
                        i++;
                    }
                }
            }
        }
        return r;
    }

    // determinant of a square matrix
    // https://www.geeksforgeeks.org/determinant-of-a-matrix/
    template<std::floating_point T, i32 n>
    constexpr T determinant(Mat<T, n, n> m)
    {
        if constexpr (n == 1)
            return m(0, 0);

        T det = 1;
        T total = 1;

        // temporary array for storing row
        T temp[n + 1]{};

        // traverse the diagonal elements
        for (i32 i = 0; i < n; i++)
        {
            i32 index = i;

            // find the index which has non zero value
            while (index < n && m(index, i) == 0)
            {
                index++;
            }

            if (index == n)
            {
                continue;
            }

            if (index != i)
            {
                // swap the diagonal element row and index row
                for (i32 j = 0; j < n; j++)
                {
                    std::swap(m(index, j), m(i, j));
                }

                // the determinant sign changes when we shift rows
                det *= (index - i) % 2 == 0 ? (T)1 : (T)(-1);
            }

            // store the diagonal row elements
            for (i32 j = 0; j < n; j++)
            {
                temp[j] = m(i, j);
            }

            // traverse every row below the diagonal element
            for (i32 j = i + 1; j < n; j++)
            {
                // value of diagonal element
                T num1 = temp[i];

                // value of next row element
                T num2 = m(j, i);

                // traverse every column of row and multiply to every row
                for (i32 k = 0; k < n; k++)
                {
                    // multiply to make the diagonal element and next row
                    // element equal
                    m(j, k) = (num1 * m(j, k)) - (num2 * temp[k]);
                }

                // Det(kA)=kDet(A);
                total *= num1;
            }
        }

        // multiply the diagonal elements to get the determinant
        for (i32 i = 0; i < n; i++)
        {
            det *= m(i, i);
        }

        // Det(kA)/k=Det(A);
        return (det / total);
    }

    // adjoint of a square matrix
    // https://www.geeksforgeeks.org/adjoint-inverse-matrix/
    template<std::floating_point T, i32 n>
    constexpr Mat<T, n, n> adjoint(const Mat<T, n, n>& m)
    {
        if constexpr (n == 1)
            return Mat<T, n, n>();

        Mat<T, n, n> r;
        for (i32 i = 0; i < n; i++)
        {
            for (i32 j = 0; j < n; j++)
            {
                // get cofactor of m[i][j]
                Mat<T, n - 1, n - 1> cf = cofactor(m, i, j);

                // sign of adj[j][i] positive if the sum of the row and column
                // indices is even.
                T sign = ((i + j) % 2 == 0) ? (T)1 : (T)(-1);

                // interchanging rows and columns to get the transpose of the
                // cofactor matrix
                r(j, i) = sign * determinant(cf);
            }
        }
        return r;
    }

    // inverted copy of a square matrix
    template<std::floating_point T, i32 n>
    constexpr Mat<T, n, n> inverse(
        const Mat<T, n, n>& m,
        bool* out_invertible = nullptr
    )
    {
        // determinant
        T det = determinant(m);
        if (det == 0)
        {
            if (out_invertible)
                *out_invertible = false;
            return Mat<T, n, n>();
        }

        // adjoint
        Mat<T, n, n> adj = adjoint(m);

        // inverse(A) = adj(A)/det(A)
        if (out_invertible)
            *out_invertible = true;
        return adj / det;
    }

    // transposed copy of a matrix
    template<typename T, i32 n_row, i32 n_col>
    constexpr Mat<T, n_col, n_row> transpose(
        const Mat<T, n_row, n_col>& m
    )
    {
        Mat<T, n_col, n_row> r;
        for (i32 row = 0; row < n_row; row++)
        {
            for (i32 col = 0; col < n_col; col++)
            {
                r(col, row) = m(row, col);
            }
        }
        return r;
    }

    using Mat1x2f = Mat<f32, 1, 2>;
    using Mat1x2d = Mat<f64, 1, 2>;
    using Mat1x2i = Mat<i32, 1, 2>;
    using Mat1x2u = Mat<u32, 1, 2>;

    using Mat2x1f = Mat<f32, 2, 1>;
    using Mat2x1d = Mat<f64, 2, 1>;
    using Mat2x1i = Mat<i32, 2, 1>;
    using Mat2x1u = Mat<u32, 2, 1>;

    using Mat1x3f = Mat<f32, 1, 3>;
    using Mat1x3d = Mat<f64, 1, 3>;
    using Mat1x3i = Mat<i32, 1, 3>;
    using Mat1x3u = Mat<u32, 1, 3>;

    using Mat3x1f = Mat<f32, 3, 1>;
    using Mat3x1d = Mat<f64, 3, 1>;
    using Mat3x1i = Mat<i32, 3, 1>;
    using Mat3x1u = Mat<u32, 3, 1>;

    using Mat1x4f = Mat<f32, 1, 4>;
    using Mat1x4d = Mat<f64, 1, 4>;
    using Mat1x4i = Mat<i32, 1, 4>;
    using Mat1x4u = Mat<u32, 1, 4>;

    using Mat4x1f = Mat<f32, 4, 1>;
    using Mat4x1d = Mat<f64, 4, 1>;
    using Mat4x1i = Mat<i32, 4, 1>;
    using Mat4x1u = Mat<u32, 4, 1>;

    using Mat2f = Mat<f32, 2, 2>;
    using Mat2d = Mat<f64, 2, 2>;
    using Mat2i = Mat<i32, 2, 2>;
    using Mat2u = Mat<u32, 2, 2>;

    using Mat2x2f = Mat<f32, 2, 2>;
    using Mat2x2d = Mat<f64, 2, 2>;
    using Mat2x2i = Mat<i32, 2, 2>;
    using Mat2x2u = Mat<u32, 2, 2>;

    using Mat2x3f = Mat<f32, 2, 3>;
    using Mat2x3d = Mat<f64, 2, 3>;
    using Mat2x3i = Mat<i32, 2, 3>;
    using Mat2x3u = Mat<u32, 2, 3>;

    using Mat3x2f = Mat<f32, 3, 2>;
    using Mat3x2d = Mat<f64, 3, 2>;
    using Mat3x2i = Mat<i32, 3, 2>;
    using Mat3x2u = Mat<u32, 3, 2>;

    using Mat2x4f = Mat<f32, 2, 4>;
    using Mat2x4d = Mat<f64, 2, 4>;
    using Mat2x4i = Mat<i32, 2, 4>;
    using Mat2x4u = Mat<u32, 2, 4>;

    using Mat4x2f = Mat<f32, 4, 2>;
    using Mat4x2d = Mat<f64, 4, 2>;
    using Mat4x2i = Mat<i32, 4, 2>;
    using Mat4x2u = Mat<u32, 4, 2>;

    using Mat3f = Mat<f32, 3, 3>;
    using Mat3d = Mat<f64, 3, 3>;
    using Mat3i = Mat<i32, 3, 3>;
    using Mat3u = Mat<u32, 3, 3>;

    using Mat3x3f = Mat<f32, 3, 3>;
    using Mat3x3d = Mat<f64, 3, 3>;
    using Mat3x3i = Mat<i32, 3, 3>;
    using Mat3x3u = Mat<u32, 3, 3>;

    using Mat3x4f = Mat<f32, 3, 4>;
    using Mat3x4d = Mat<f64, 3, 4>;
    using Mat3x4i = Mat<i32, 3, 4>;
    using Mat3x4u = Mat<u32, 3, 4>;

    using Mat4x3f = Mat<f32, 4, 3>;
    using Mat4x3d = Mat<f64, 4, 3>;
    using Mat4x3i = Mat<i32, 4, 3>;
    using Mat4x3u = Mat<u32, 4, 3>;

    using Mat4f = Mat<f32, 4, 4>;
    using Mat4d = Mat<f64, 4, 4>;
    using Mat4i = Mat<i32, 4, 4>;
    using Mat4u = Mat<u32, 4, 4>;

    using Mat4x4f = Mat<f32, 4, 4>;
    using Mat4x4d = Mat<f64, 4, 4>;
    using Mat4x4i = Mat<i32, 4, 4>;
    using Mat4x4u = Mat<u32, 4, 4>;

    // MARK: vectors

    template<typename T>
    class Vec2
    {
    public:
        T x = 0, y = 0;

        constexpr Vec2() = default;

        constexpr Vec2(T x, T y)
            : x(x), y(y)
        {}

        constexpr Vec2(T s)
            : x(s), y(s)
        {}

        template<typename U>
        explicit constexpr Vec2(const Vec2<U>& v)
            : x((T)v.x), y((T)v.y)
        {}

        explicit constexpr Vec2(const Mat<T, 1, 2>& m)
            : x(m(0)), y(m(1))
        {}

        explicit constexpr Vec2(const Mat<T, 2, 1>& m)
            : x(m(0)), y(m(1))
        {}

        explicit constexpr operator Mat<T, 1, 2>() const
        {
            return Mat<T, 1, 2>({ x, y });
        }

        explicit constexpr operator Mat<T, 2, 1>() const
        {
            return Mat<T, 2, 1>({ x, y });
        }

        std::string to_string() const
        {
            return std::string('[')
                + to_string(x) + ", "
                + to_string(y)
                + ']';
        }

        friend std::ostream& operator<<(std::ostream& os, const Vec2<T>& v)
        {
            os << v.to_string();
            return os;
        }

        constexpr Vec2<T> operator+(const Vec2<T>& v) const
        {
            return Vec2<T>(x + v.x, y + v.y);
        }

        constexpr Vec2<T>& operator+=(const Vec2<T>& v)
        {
            x += v.x;
            y += v.y;
            return *this;
        }

        constexpr Vec2<T> operator-(const Vec2<T>& v) const
        {
            return Vec2<T>(x - v.x, y - v.y);
        }

        constexpr Vec2<T>& operator-=(const Vec2<T>& v)
        {
            x -= v.x;
            y -= v.y;
            return *this;
        }

        constexpr Vec2<T> operator*(T s) const
        {
            return Vec2<T>(s * x, s * y);
        }

        constexpr Vec2<T>& operator*=(T s)
        {
            x *= s;
            y *= s;
            return *this;
        }

        constexpr Vec2<T> operator*(Vec2<T> v) const
        {
            return Vec2<T>(x * v.x, y * v.y);
        }

        constexpr Vec2<T>& operator*=(Vec2<T> v)
        {
            x *= v.x;
            y *= v.y;
            return *this;
        }

        constexpr Vec2<T> operator/(T s) const
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                T inv = (T)1 / s;
                return Vec2<T>(x * inv, y * inv);
            }
            else
            {
                return Vec2<T>(x / s, y / s);
            }
        }

        constexpr Vec2<T>& operator/=(T s)
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                T inv = (T)1 / s;
                x *= inv;
                y *= inv;
            }
            else
            {
                x /= s;
                y /= s;
            }
            return *this;
        }

        constexpr Vec2<T> operator/(Vec2<T> v) const
        {
            return Vec2<T>(x / v.x, y / v.y);
        }

        constexpr Vec2<T>& operator/=(Vec2<T> v)
        {
            x /= v.x;
            y /= v.y;
            return *this;
        }

        template<typename = std::enable_if_t<std::is_integral_v<T>>>
        constexpr Vec2<T> operator%(Vec2<T> v) const
        {
            return Vec2<T>(x % v.x, y % v.y);
        }

        template<typename = std::enable_if_t<std::is_integral_v<T>>>
        constexpr Vec2<T>& operator%=(Vec2<T> v)
        {
            x %= v.x;
            y %= v.y;
            return *this;
        }

        constexpr bool operator==(const Vec2<T>& v) const
        {
            return x == v.x && y == v.y;
        }

        constexpr bool operator!=(const Vec2<T>& v) const
        {
            return x != v.x || y != v.y;
        }

        constexpr Vec2<T> operator-() const
        {
            return Vec2<T>(-x, -y);
        }

        constexpr T operator[](i32 i) const
        {
            if (i == 0) return x;
            return y;
        }

        constexpr T& operator[](i32 i)
        {
            if (i == 0) return x;
            return y;
        }

        constexpr Vec2<T> permute(i32 x, i32 y) const
        {
            return Vec2<T>((*this)[x], (*this)[y]);
        }

        constexpr Vec2<T> yx() const
        {
            return Vec2<T>(y, x);
        }

        constexpr i32 n_components() const
        {
            return 2;
        }

        constexpr Mat<T, 1, 2> as_row() const
        {
            return Mat<T, 1, 2>(&x);
        }

        constexpr Mat<T, 2, 1> as_col() const
        {
            return Mat<T, 2, 1>(&x);
        }

    };

    template<typename T>
    constexpr Vec2<T> operator+(T s, const Vec2<T>& v)
    {
        return v + s;
    }

    template<typename T>
    constexpr Vec2<T> operator-(T s, const Vec2<T>& v)
    {
        return (-v) + s;
    }

    template<typename T>
    constexpr Vec2<T> operator*(T s, const Vec2<T>& v)
    {
        return v * s;
    }

    template<typename T>
    constexpr Vec2<T> operator/(T s, const Vec2<T>& v)
    {
        return Vec2<T>(s / v.x, s / v.y);
    }

    template<std::floating_point T>
    constexpr Vec2<T> radians(const Vec2<T>& degrees)
    {
        return degrees * deg2rad;
    }

    template<std::floating_point T>
    constexpr Vec2<T> degrees(const Vec2<T>& radians)
    {
        return radians * rad2deg;
    }

    template<std::floating_point T>
    inline Vec2<T> sin(const Vec2<T>& v)
    {
        return Vec2<T>(sin(v.x), sin(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> cos(const Vec2<T>& v)
    {
        return Vec2<T>(cos(v.x), cos(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> tan(const Vec2<T>& v)
    {
        return Vec2<T>(tan(v.x), tan(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> asin(const Vec2<T>& v)
    {
        return Vec2<T>(asin(v.x), asin(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> acos(const Vec2<T>& v)
    {
        return Vec2<T>(acos(v.x), acos(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> atan(const Vec2<T>& v)
    {
        return Vec2<T>(atan(v.x), atan(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> sinh(const Vec2<T>& v)
    {
        return Vec2<T>(sinh(v.x), sinh(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> cosh(const Vec2<T>& v)
    {
        return Vec2<T>(cosh(v.x), cosh(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> tanh(const Vec2<T>& v)
    {
        return Vec2<T>(tanh(v.x), tanh(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> asinh(const Vec2<T>& v)
    {
        return Vec2<T>(asinh(v.x), asinh(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> acosh(const Vec2<T>& v)
    {
        return Vec2<T>(acosh(v.x), acosh(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> atanh(const Vec2<T>& v)
    {
        return Vec2<T>(atanh(v.x), atanh(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> pow(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>(pow(v1.x, v2.x), pow(v1.y, v2.y));
    }

    template<std::floating_point T>
    inline Vec2<T> pow(const Vec2<T>& v1, T v2)
    {
        return Vec2<T>(pow(v1.x, v2), pow(v1.y, v2));
    }

    template<std::floating_point T>
    inline Vec2<T> exp(const Vec2<T>& v)
    {
        return Vec2<T>(exp(v.x), exp(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> log(const Vec2<T>& v)
    {
        return Vec2<T>(log(v.x), log(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> exp2(const Vec2<T>& v)
    {
        return Vec2<T>(exp2(v.x), exp2(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> log2(const Vec2<T>& v)
    {
        return Vec2<T>(log2(v.x), log2(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> sqrt(const Vec2<T>& v)
    {
        return Vec2<T>(sqrt(v.x), sqrt(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> inversesqrt(const Vec2<T>& v)
    {
        return Vec2<T>(inversesqrt(v.x), inversesqrt(v.y));
    }

    template<typename T>
    inline Vec2<T> abs(const Vec2<T>& v)
    {
        return Vec2<T>(abs(v.x), abs(v.y));
    }

    template<typename T>
    constexpr Vec2<T> sign(const Vec2<T>& v)
    {
        return Vec2<T>(sign(v.x), sign(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> floor(const Vec2<T>& v)
    {
        return Vec2<T>(floor(v.x), floor(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> ceil(const Vec2<T>& v)
    {
        return Vec2<T>(ceil(v.x), ceil(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> trunc(const Vec2<T>& v)
    {
        return Vec2<T>(trunc(v.x), trunc(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> fract(const Vec2<T>& v)
    {
        return Vec2<T>(fract(v.x), fract(v.y));
    }

    template<std::floating_point T>
    inline Vec2<T> mod(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>(mod(v1.x, v2.x), mod(v1.y, v2.y));
    }

    template<std::floating_point T>
    inline Vec2<T> mod(const Vec2<T>& v1, T v2)
    {
        return Vec2<T>(mod(v1.x, v2), mod(v1.y, v2));
    }

    template<std::floating_point T>
    inline Vec2<T> modf(const Vec2<T>& v, Vec2<T>& i)
    {
        return Vec2<T>(modf(v.x, i.x), modf(v.y, i.y));
    }

    template<std::floating_point T>
    inline Vec2<T> wrap(const Vec2<T>& v, T start, T end)
    {
        return start + mod(v - start, end - start);
    }

    template<typename T>
    constexpr Vec2<T> min(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>(
            min(v1.x, v2.x),
            min(v1.y, v2.y)
        );
    }

    template<typename T>
    constexpr Vec2<T> min(const Vec2<T>& v1, T v2)
    {
        return Vec2<T>(
            min(v1.x, v2),
            min(v1.y, v2)
        );
    }

    template<typename T>
    constexpr Vec2<T> min(T v1, const Vec2<T>& v2)
    {
        return Vec2<T>(
            min(v1, v2.x),
            min(v1, v2.y)
        );
    }

    template<typename T>
    constexpr Vec2<T> max(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>(
            max(v1.x, v2.x),
            max(v1.y, v2.y)
        );
    }

    template<typename T>
    constexpr Vec2<T> max(const Vec2<T>& v1, T v2)
    {
        return Vec2<T>(
            max(v1.x, v2),
            max(v1.y, v2)
        );
    }

    template<typename T>
    constexpr Vec2<T> max(T v1, const Vec2<T>& v2)
    {
        return Vec2<T>(
            max(v1, v2.x),
            max(v1, v2.y)
        );
    }

    template<typename T>
    constexpr Vec2<T> clamp(const Vec2<T>& v, T min, T max)
    {
        return Vec2<T>(clamp(v.x, min, max), clamp(v.y, min, max));
    }

    template<typename T>
    constexpr Vec2<T> clamp(
        const Vec2<T>& v,
        const Vec2<T>& min,
        const Vec2<T>& max
    )
    {
        return Vec2<T>(
            clamp(v.x, min.x, max.x),
            clamp(v.y, min.y, max.y)
        );
    }

    template<std::floating_point T>
    constexpr Vec2<T> clamp01(const Vec2<T>& v)
    {
        return Vec2<T>(clamp01(v.x), clamp01(v.y));
    }

    template<std::floating_point T>
    constexpr Vec2<T> mix(
        const Vec2<T>& v1,
        const Vec2<T>& v2,
        T a
    )
    {
        return v1 + a * (v2 - v1);
    }

    template<std::floating_point T>
    constexpr Vec2<T> remap(
        const Vec2<T>& v,
        T a_start,
        T a_end,
        T b_start,
        T b_end
    )
    {
        return
            b_start + ((b_end - b_start) / (a_end - a_start)) * (v - a_start);
    }

    template<std::floating_point T>
    constexpr Vec2<T> remap_clamp(
        const Vec2<T>& v,
        T a_start,
        T a_end,
        T b_start,
        T b_end
    )
    {
        Vec2<T> t = clamp01((v - a_start) / (a_end - a_start));
        return b_start + t * (b_end - b_start);
    }

    template<std::floating_point T>
    constexpr Vec2<T> remap01(
        const Vec2<T>& v,
        T a_start,
        T a_end
    )
    {
        return clamp01((v - a_start) / (a_end - a_start));
    }

    template<typename T>
    constexpr Vec2<T> step(T edge, const Vec2<T>& v)
    {
        return Vec2<T>(step(edge, v.x), step(edge, v.y));
    }

    template<std::floating_point T>
    constexpr Vec2<T> smoothstep(
        T edge0,
        T edge1,
        const Vec2<T>& v
    )
    {
        return Vec2<T>(
            smoothstep(edge0, edge1, v.x), smoothstep(edge0, edge1, v.y)
        );
    }

    template<typename T>
    constexpr T length_squared(const Vec2<T>& v)
    {
        return v.x * v.x + v.y * v.y;
    }

    template<std::floating_point T>
    inline T length(const Vec2<T>& v)
    {
        return sqrt(length_squared(v));
    }

    template<typename T>
    constexpr T distance_squared(
        const Vec2<T>& v1,
        const Vec2<T>& v2
    )
    {
        return length_squared(v1 - v2);
    }

    template<std::floating_point T>
    inline T distance(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return length(v1 - v2);
    }

    template<typename T>
    constexpr T dot(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return v1.x * v2.x + v1.y * v2.y;
    }

    template<std::floating_point T>
    inline Vec2<T> normalize(const Vec2<T>& v)
    {
        return v / length(v);
    }

    template<std::floating_point T>
    constexpr Vec2<T> faceforward(
        const Vec2<T>& n,
        const Vec2<T>& i,
        const Vec2<T>& nref
    )
    {
        if (dot(nref, i) < (T)0)
            return n;
        return -n;
    }

    template<std::floating_point T>
    constexpr Vec2<T> reflect(const Vec2<T>& i, const Vec2<T>& n)
    {
        return i - (T)2 * dot(n, i) * n;
    }

    template<std::floating_point T>
    inline Vec2<T> refract(
        const Vec2<T>& i,
        const Vec2<T>& n,
        T eta
    )
    {
        T dp = dot(n, i);
        T k = (T)1 - eta * eta * ((T)1 - dp * dp);

        if (k < (T)0)
            return (T)0;

        return eta * i - (eta * dp + sqrt(k)) * n;
    }

    template<typename T>
    constexpr T min_component(const Vec2<T>& v)
    {
        return min(v.x, v.y);
    }

    template<typename T>
    constexpr T max_component(const Vec2<T>& v)
    {
        return max(v.x, v.y);
    }

    template<typename T>
    constexpr i32 min_component_index(const Vec2<T>& v)
    {
        return (v.x < v.y) ? 0 : 1;
    }

    template<typename T>
    constexpr i32 max_component_index(const Vec2<T>& v)
    {
        return (v.x > v.y) ? 0 : 1;
    }

    using Vec2f = Vec2<f32>;
    using Vec2d = Vec2<f64>;
    using Vec2i = Vec2<i32>;
    using Vec2u = Vec2<i64>;

    template<typename T>
    class Vec3
    {
    public:
        T x = 0, y = 0, z = 0;

        constexpr Vec3() = default;

        constexpr Vec3(T x, T y, T z)
            : x(x), y(y), z(z)
        {}

        constexpr Vec3(T s)
            : x(s), y(s), z(s)
        {}

        constexpr Vec3(Vec2<T> xy, T z)
            : x(xy.x), y(xy.y), z(z)
        {}

        constexpr Vec3(T x, Vec2<T> yz)
            : x(x), y(yz.x), z(yz.y)
        {}

        template<typename U>
        explicit constexpr Vec3(const Vec3<U>& v)
            : x((T)v.x), y((T)v.y), z((T)v.z)
        {}

        explicit constexpr Vec3(const Mat<T, 1, 3>& m)
            : x(m(0)), y(m(1)), z(m(2))
        {}

        explicit constexpr Vec3(const Mat<T, 3, 1>& m)
            : x(m(0)), y(m(1)), z(m(2))
        {}

        explicit constexpr operator Mat<T, 1, 3>() const
        {
            return Mat<T, 1, 3>({ x, y, z });
        }

        explicit constexpr operator Mat<T, 3, 1>() const
        {
            return Mat<T, 3, 1>({ x, y, z });
        }

        std::string to_string() const
        {
            return std::string('[')
                + to_string(x) + ", "
                + to_string(y) + ", "
                + to_string(z)
                + ']';
        }

        friend std::ostream& operator<<(std::ostream& os, const Vec3<T>& v)
        {
            os << v.to_string();
            return os;
        }

        constexpr Vec3<T> operator+(const Vec3<T>& v) const
        {
            return Vec3<T>(x + v.x, y + v.y, z + v.z);
        }

        constexpr Vec3<T>& operator+=(const Vec3<T>& v)
        {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        constexpr Vec3<T> operator-(const Vec3<T>& v) const
        {
            return Vec3<T>(x - v.x, y - v.y, z - v.z);
        }

        constexpr Vec3<T>& operator-=(const Vec3<T>& v)
        {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        constexpr Vec3<T> operator*(T s) const
        {
            return Vec3<T>(s * x, s * y, s * z);
        }

        constexpr Vec3<T>& operator*=(T s)
        {
            x *= s;
            y *= s;
            z *= s;
            return *this;
        }

        constexpr Vec3<T> operator*(Vec3<T> v) const
        {
            return Vec3<T>(x * v.x, y * v.y, z * v.z);
        }

        constexpr Vec3<T>& operator*=(Vec3<T> v)
        {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            return *this;
        }

        constexpr Vec3<T> operator/(T s) const
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                T inv = (T)1 / s;
                return Vec3<T>(x * inv, y * inv, z * inv);
            }
            else
            {
                return Vec3<T>(x / s, y / s, z / s);
            }
        }

        constexpr Vec3<T>& operator/=(T s)
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                T inv = (T)1 / s;
                x *= inv;
                y *= inv;
                z *= inv;
            }
            else
            {
                x /= s;
                y /= s;
                z /= s;
            }
            return *this;
        }

        constexpr Vec3<T> operator/(Vec3<T> v) const
        {
            return Vec3<T>(x / v.x, y / v.y, z / v.z);
        }

        constexpr Vec3<T>& operator/=(Vec3<T> v)
        {
            x /= v.x;
            y /= v.y;
            z /= v.z;
            return *this;
        }

        template<typename = std::enable_if_t<std::is_integral_v<T>>>
        constexpr Vec3<T> operator%(Vec3<T> v) const
        {
            return Vec3<T>(x % v.x, y % v.y, z % v.z);
        }

        template<typename = std::enable_if_t<std::is_integral_v<T>>>
        constexpr Vec3<T>& operator%=(Vec3<T> v)
        {
            x %= v.x;
            y %= v.y;
            z %= v.z;
            return *this;
        }

        constexpr bool operator==(const Vec3<T>& v) const
        {
            return x == v.x && y == v.y && z == v.z;
        }

        constexpr bool operator!=(const Vec3<T>& v) const
        {
            return x != v.x || y != v.y || z != v.z;
        }

        constexpr Vec3<T> operator-() const
        {
            return Vec3<T>(-x, -y, -z);
        }

        constexpr T operator[](i32 i) const
        {
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }

        constexpr T& operator[](i32 i)
        {
            if (i == 0) return x;
            if (i == 1) return y;
            return z;
        }

        constexpr Vec2<T> permute(i32 x, i32 y) const
        {
            return Vec2<T>((*this)[x], (*this)[y]);
        }

        constexpr Vec3<T> permute(i32 x, i32 y, i32 z) const
        {
            return Vec3<T>((*this)[x], (*this)[y], (*this)[z]);
        }

        constexpr i32 n_components() const
        {
            return 3;
        }

        constexpr Mat<T, 1, 3> as_row() const
        {
            return Mat<T, 1, 3>(&x);
        }

        constexpr Mat<T, 3, 1> as_col() const
        {
            return Mat<T, 3, 1>(&x);
        }

    };

    template<typename T>
    constexpr Vec3<T> operator+(T s, const Vec3<T>& v)
    {
        return v + s;
    }

    template<typename T>
    constexpr Vec3<T> operator-(T s, const Vec3<T>& v)
    {
        return (-v) + s;
    }

    template<typename T>
    constexpr Vec3<T> operator*(T s, const Vec3<T>& v)
    {
        return v * s;
    }

    template<typename T>
    constexpr Vec3<T> operator/(T s, const Vec3<T>& v)
    {
        return Vec3<T>(s / v.x, s / v.y, s / v.z);
    }

    template<std::floating_point T>
    constexpr Vec3<T> radians(const Vec3<T>& degrees)
    {
        return degrees * deg2rad;
    }

    template<std::floating_point T>
    constexpr Vec3<T> degrees(const Vec3<T>& radians)
    {
        return radians * rad2deg;
    }

    template<std::floating_point T>
    inline Vec3<T> sin(const Vec3<T>& v)
    {
        return Vec3<T>(sin(v.x), sin(v.y), sin(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> cos(const Vec3<T>& v)
    {
        return Vec3<T>(cos(v.x), cos(v.y), cos(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> tan(const Vec3<T>& v)
    {
        return Vec3<T>(tan(v.x), tan(v.y), tan(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> asin(const Vec3<T>& v)
    {
        return Vec3<T>(asin(v.x), asin(v.y), asin(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> acos(const Vec3<T>& v)
    {
        return Vec3<T>(acos(v.x), acos(v.y), acos(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> atan(const Vec3<T>& v)
    {
        return Vec3<T>(atan(v.x), atan(v.y), atan(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> sinh(const Vec3<T>& v)
    {
        return Vec3<T>(sinh(v.x), sinh(v.y), sinh(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> cosh(const Vec3<T>& v)
    {
        return Vec3<T>(cosh(v.x), cosh(v.y), cosh(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> tanh(const Vec3<T>& v)
    {
        return Vec3<T>(tanh(v.x), tanh(v.y), tanh(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> asinh(const Vec3<T>& v)
    {
        return Vec3<T>(asinh(v.x), asinh(v.y), asinh(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> acosh(const Vec3<T>& v)
    {
        return Vec3<T>(acosh(v.x), acosh(v.y), acosh(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> atanh(const Vec3<T>& v)
    {
        return Vec3<T>(atanh(v.x), atanh(v.y), atanh(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> pow(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>(pow(v1.x, v2.x), pow(v1.y, v2.y), pow(v1.z, v2.z));
    }

    template<std::floating_point T>
    inline Vec3<T> pow(const Vec3<T>& v1, T v2)
    {
        return Vec3<T>(pow(v1.x, v2), pow(v1.y, v2), pow(v1.z, v2));
    }

    template<std::floating_point T>
    inline Vec3<T> exp(const Vec3<T>& v)
    {
        return Vec3<T>(exp(v.x), exp(v.y), exp(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> log(const Vec3<T>& v)
    {
        return Vec3<T>(log(v.x), log(v.y), log(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> exp2(const Vec3<T>& v)
    {
        return Vec3<T>(exp2(v.x), exp2(v.y), exp2(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> log2(const Vec3<T>& v)
    {
        return Vec3<T>(log2(v.x), log2(v.y), log2(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> sqrt(const Vec3<T>& v)
    {
        return Vec3<T>(sqrt(v.x), sqrt(v.y), sqrt(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> inversesqrt(const Vec3<T>& v)
    {
        return Vec3<T>(
            inversesqrt(v.x), inversesqrt(v.y), inversesqrt(v.z)
        );
    }

    template<typename T>
    inline Vec3<T> abs(const Vec3<T>& v)
    {
        return Vec3<T>(abs(v.x), abs(v.y), abs(v.z));
    }

    template<typename T>
    constexpr Vec3<T> sign(const Vec3<T>& v)
    {
        return Vec3<T>(sign(v.x), sign(v.y), sign(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> floor(const Vec3<T>& v)
    {
        return Vec3<T>(floor(v.x), floor(v.y), floor(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> ceil(const Vec3<T>& v)
    {
        return Vec3<T>(ceil(v.x), ceil(v.y), ceil(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> trunc(const Vec3<T>& v)
    {
        return Vec3<T>(trunc(v.x), trunc(v.y), trunc(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> fract(const Vec3<T>& v)
    {
        return Vec3<T>(fract(v.x), fract(v.y), fract(v.z));
    }

    template<std::floating_point T>
    inline Vec3<T> mod(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>(mod(v1.x, v2.x), mod(v1.y, v2.y), mod(v1.z, v2.z));
    }

    template<std::floating_point T>
    inline Vec3<T> mod(const Vec3<T>& v1, T v2)
    {
        return Vec3<T>(mod(v1.x, v2), mod(v1.y, v2), mod(v1.z, v2));
    }

    template<std::floating_point T>
    inline Vec3<T> modf(const Vec3<T>& v, Vec3<T>& i)
    {
        return Vec3<T>(modf(v.x, i.x), modf(v.y, i.y), modf(v.z, i.z));
    }

    template<std::floating_point T>
    inline Vec3<T> wrap(const Vec3<T>& v, T start, T end)
    {
        return start + mod(v - start, end - start);
    }

    template<typename T>
    constexpr Vec3<T> min(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>(
            min(v1.x, v2.x),
            min(v1.y, v2.y),
            min(v1.z, v2.z)
        );
    }

    template<typename T>
    constexpr Vec3<T> min(const Vec3<T>& v1, T v2)
    {
        return Vec3<T>(
            min(v1.x, v2),
            min(v1.y, v2),
            min(v1.z, v2)
        );
    }

    template<typename T>
    constexpr Vec3<T> min(T v1, const Vec3<T>& v2)
    {
        return Vec3<T>(
            min(v1, v2.x),
            min(v1, v2.y),
            min(v1, v2.z)
        );
    }

    template<typename T>
    constexpr Vec3<T> max(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>(
            max(v1.x, v2.x),
            max(v1.y, v2.y),
            max(v1.z, v2.z)
        );
    }

    template<typename T>
    constexpr Vec3<T> max(const Vec3<T>& v1, T v2)
    {
        return Vec3<T>(
            max(v1.x, v2),
            max(v1.y, v2),
            max(v1.z, v2)
        );
    }

    template<typename T>
    constexpr Vec3<T> max(T v1, const Vec3<T>& v2)
    {
        return Vec3<T>(
            max(v1, v2.x),
            max(v1, v2.y),
            max(v1, v2.z)
        );
    }


    template<typename T>
    constexpr Vec3<T> clamp(const Vec3<T>& v, T min, T max)
    {
        return Vec3<T>(
            clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max)
        );
    }

    template<typename T>
    constexpr Vec3<T> clamp(
        const Vec3<T>& v,
        const Vec3<T>& min,
        const Vec3<T>& max
    )
    {
        return Vec3<T>(
            clamp(v.x, min.x, max.x),
            clamp(v.y, min.y, max.y),
            clamp(v.z, min.z, max.z)
        );
    }

    template<std::floating_point T>
    constexpr Vec3<T> clamp01(const Vec3<T>& v)
    {
        return Vec3<T>(clamp01(v.x), clamp01(v.y), clamp01(v.z));
    }

    template<std::floating_point T>
    constexpr Vec3<T> mix(
        const Vec3<T>& v1,
        const Vec3<T>& v2,
        T a
    )
    {
        return v1 + a * (v2 - v1);
    }

    template<std::floating_point T>
    constexpr Vec3<T> remap(
        const Vec3<T>& v,
        T a_start,
        T a_end,
        T b_start,
        T b_end
    )
    {
        return
            b_start + ((b_end - b_start) / (a_end - a_start)) * (v - a_start);
    }

    template<std::floating_point T>
    constexpr Vec3<T> remap_clamp(
        const Vec3<T>& v,
        T a_start,
        T a_end,
        T b_start,
        T b_end
    )
    {
        Vec3<T> t = clamp01((v - a_start) / (a_end - a_start));
        return b_start + t * (b_end - b_start);
    }

    template<std::floating_point T>
    constexpr Vec3<T> remap01(
        const Vec3<T>& v,
        T a_start,
        T a_end
    )
    {
        return clamp01((v - a_start) / (a_end - a_start));
    }

    template<typename T>
    constexpr Vec3<T> step(T edge, const Vec3<T>& v)
    {
        return Vec3<T>(step(edge, v.x), step(edge, v.y), step(edge, v.z));
    }

    template<std::floating_point T>
    constexpr Vec3<T> smoothstep(
        T edge0,
        T edge1,
        const Vec3<T>& v
    )
    {
        return Vec3<T>(
            smoothstep(edge0, edge1, v.x),
            smoothstep(edge0, edge1, v.y),
            smoothstep(edge0, edge1, v.z)
        );
    }

    template<typename T>
    constexpr T length_squared(const Vec3<T>& v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    template<std::floating_point T>
    inline T length(const Vec3<T>& v)
    {
        return sqrt(length_squared(v));
    }

    template<typename T>
    constexpr T distance_squared(
        const Vec3<T>& v1,
        const Vec3<T>& v2
    )
    {
        return length_squared(v1 - v2);
    }

    template<std::floating_point T>
    inline T distance(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return length(v1 - v2);
    }

    template<typename T>
    constexpr T dot(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    template<typename T>
    constexpr Vec3<T> cross(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>(
            (v1.y * v2.z) - (v1.z * v2.y),
            (v1.z * v2.x) - (v1.x * v2.z),
            (v1.x * v2.y) - (v1.y * v2.x)
        );
    }

    template<std::floating_point T>
    inline Vec3<T> normalize(const Vec3<T>& v)
    {
        return v / length(v);
    }

    template<std::floating_point T>
    constexpr Vec3<T> faceforward(
        const Vec3<T>& n,
        const Vec3<T>& i,
        const Vec3<T>& nref
    )
    {
        if (dot(nref, i) < (T)0)
            return n;
        return -n;
    }

    template<std::floating_point T>
    constexpr Vec3<T> reflect(const Vec3<T>& i, const Vec3<T>& n)
    {
        return i - (T)2 * dot(n, i) * n;
    }

    template<std::floating_point T>
    inline Vec3<T> refract(
        const Vec3<T>& i,
        const Vec3<T>& n,
        T eta
    )
    {
        T dp = dot(n, i);
        T k = (T)1 - eta * eta * ((T)1 - dp * dp);

        if (k < (T)0)
            return (T)0;

        return eta * i - (eta * dp + sqrt(k)) * n;
    }

    template<typename T>
    constexpr T min_component(const Vec3<T>& v)
    {
        return min(v.x, min(v.y, v.z));
    }

    template<typename T>
    constexpr T max_component(const Vec3<T>& v)
    {
        return max(v.x, max(v.y, v.z));
    }

    template<typename T>
    constexpr i32 min_component_index(const Vec3<T>& v)
    {
        return (v.x < v.y)
            ? ((v.x < v.z) ? 0 : 2)
            : ((v.y < v.z) ? 1 : 2);
    }

    template<typename T>
    constexpr i32 max_component_index(const Vec3<T>& v)
    {
        return (v.x > v.y)
            ? ((v.x > v.z) ? 0 : 2)
            : ((v.y > v.z) ? 1 : 2);
    }

    using Vec3f = Vec3<f32>;
    using Vec3d = Vec3<f64>;
    using Vec3i = Vec3<i32>;
    using Vec3u = Vec3<u32>;

    template<typename T>
    class Vec4
    {
    public:
        T x = 0, y = 0, z = 0, w = 0;

        constexpr Vec4() = default;

        constexpr Vec4(T x, T y, T z, T w)
            : x(x), y(y), z(z), w(w)
        {}

        constexpr Vec4(T s)
            : x(s), y(s), z(s), w(s)
        {}

        constexpr Vec4(Vec2<T> xy, T z, T w)
            : x(xy.x), y(xy.y), z(z), w(w)
        {}

        constexpr Vec4(T x, Vec2<T> yz, T w)
            : x(x), y(yz.x), z(yz.y), w(w)
        {}

        constexpr Vec4(T x, T y, Vec2<T> zw)
            : x(x), y(y), z(zw.x), w(zw.y)
        {}

        constexpr Vec4(Vec2<T> xy, Vec2<T> zw)
            : x(xy.x), y(xy.y), z(zw.x), w(zw.y)
        {}

        constexpr Vec4(Vec3<T> xyz, T w)
            : x(xyz.x), y(xyz.y), z(xyz.z), w(w)
        {}

        constexpr Vec4(T x, Vec3<T> yzw)
            : x(x), y(yzw.x), z(yzw.y), w(yzw.z)
        {}

        template<typename U>
        explicit constexpr Vec4(const Vec4<U>& v)
            : x((T)v.x), y((T)v.y), z((T)v.z), w((T)v.w)
        {}

        explicit constexpr Vec4(const Mat<T, 1, 4>& m)
            : x(m(0)), y(m(1)), z(m(2)), w(m(3))
        {}

        explicit constexpr Vec4(const Mat<T, 4, 1>& m)
            : x(m(0)), y(m(1)), z(m(2)), w(m(3))
        {}

        explicit constexpr operator Mat<T, 1, 4>() const
        {
            return Mat<T, 1, 4>({ x, y, z, w });
        }

        explicit constexpr operator Mat<T, 4, 1>() const
        {
            return Mat<T, 4, 1>({ x, y, z, w });
        }

        std::string to_string() const
        {
            return std::string('[')
                + to_string(x) + ", "
                + to_string(y) + ", "
                + to_string(z) + ", "
                + to_string(w)
                + ']';
        }

        friend std::ostream& operator<<(std::ostream& os, const Vec4<T>& v)
        {
            os << v.to_string();
            return os;
        }

        constexpr Vec4<T> operator+(const Vec4<T>& v) const
        {
            return Vec4<T>(x + v.x, y + v.y, z + v.z, w + v.w);
        }

        constexpr Vec4<T>& operator+=(const Vec4<T>& v)
        {
            x += v.x;
            y += v.y;
            z += v.z;
            w += v.w;
            return *this;
        }

        constexpr Vec4<T> operator-(const Vec4<T>& v) const
        {
            return Vec4<T>(x - v.x, y - v.y, z - v.z, w - v.w);
        }

        constexpr Vec4<T>& operator-=(const Vec4<T>& v)
        {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            w -= v.w;
            return *this;
        }

        constexpr Vec4<T> operator*(T s) const
        {
            return Vec4<T>(s * x, s * y, s * z, s * w);
        }

        constexpr Vec4<T>& operator*=(T s)
        {
            x *= s;
            y *= s;
            z *= s;
            w *= s;
            return *this;
        }

        constexpr Vec4<T> operator*(Vec4<T> v) const
        {
            return Vec4<T>(x * v.x, y * v.y, z * v.z, w * v.w);
        }

        constexpr Vec4<T>& operator*=(Vec4<T> v)
        {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            w *= v.w;
            return *this;
        }

        constexpr Vec4<T> operator/(T s) const
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                T inv = (T)1 / s;
                return Vec4<T>(x * inv, y * inv, z * inv, w * inv);
            }
            else
            {
                return Vec4<T>(x / s, y / s, z / s, w / s);
            }
        }

        constexpr Vec4<T>& operator/=(T s)
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                T inv = (T)1 / s;
                x *= inv;
                y *= inv;
                z *= inv;
                w *= inv;
            }
            else
            {
                x /= s;
                y /= s;
                z /= s;
                w /= s;
            }
            return *this;
        }

        constexpr Vec4<T> operator/(Vec4<T> v) const
        {
            return Vec4<T>(x / v.x, y / v.y, z / v.z, w / v.w);
        }

        constexpr Vec4<T>& operator/=(Vec4<T> v)
        {
            x /= v.x;
            y /= v.y;
            z /= v.z;
            w /= v.w;
            return *this;
        }

        template<typename = std::enable_if_t<std::is_integral_v<T>>>
        constexpr Vec4<T> operator%(Vec4<T> v) const
        {
            return Vec4<T>(x % v.x, y % v.y, z % v.z, w % v.w);
        }

        template<typename = std::enable_if_t<std::is_integral_v<T>>>
        constexpr Vec4<T>& operator%=(Vec4<T> v)
        {
            x %= v.x;
            y %= v.y;
            z %= v.z;
            w %= v.w;
            return *this;
        }

        constexpr bool operator==(const Vec4<T>& v) const
        {
            return x == v.x && y == v.y && z == v.z && w == v.w;
        }

        constexpr bool operator!=(const Vec4<T>& v) const
        {
            return x != v.x || y != v.y || z != v.z || w != v.w;
        }

        constexpr Vec4<T> operator-() const
        {
            return Vec4<T>(-x, -y, -z, -w);
        }

        constexpr T operator[](i32 i) const
        {
            if (i == 0) return x;
            if (i == 1) return y;
            if (i == 2) return z;
            return w;
        }

        constexpr T& operator[](i32 i)
        {
            if (i == 0) return x;
            if (i == 1) return y;
            if (i == 2) return z;
            return w;
        }

        constexpr Vec2<T> permute(i32 x, i32 y) const
        {
            return Vec2<T>((*this)[x], (*this)[y]);
        }

        constexpr Vec3<T> permute(i32 x, i32 y, i32 z) const
        {
            return Vec3<T>((*this)[x], (*this)[y], (*this)[z]);
        }

        constexpr Vec4<T> permute(i32 x, i32 y, i32 z, i32 w) const
        {
            return Vec4<T>((*this)[x], (*this)[y], (*this)[z], (*this)[w]);
        }

        constexpr i32 n_components() const
        {
            return 4;
        }

        constexpr Mat<T, 1, 4> as_row() const
        {
            return Mat<T, 1, 4>(&x);
        }

        constexpr Mat<T, 4, 1> as_col() const
        {
            return Mat<T, 4, 1>(&x);
        }

    };

    template<typename T>
    constexpr Vec4<T> operator+(T s, const Vec4<T>& v)
    {
        return v + s;
    }

    template<typename T>
    constexpr Vec4<T> operator-(T s, const Vec4<T>& v)
    {
        return (-v) + s;
    }

    template<typename T>
    constexpr Vec4<T> operator*(T s, const Vec4<T>& v)
    {
        return v * s;
    }

    template<typename T>
    constexpr Vec4<T> operator/(T s, const Vec4<T>& v)
    {
        return Vec4<T>(s / v.x, s / v.y, s / v.z, s / v.w);
    }

    template<std::floating_point T>
    constexpr Vec4<T> radians(const Vec4<T>& degrees)
    {
        return degrees * deg2rad;
    }

    template<std::floating_point T>
    constexpr Vec4<T> degrees(const Vec4<T>& radians)
    {
        return radians * rad2deg;
    }

    template<std::floating_point T>
    inline Vec4<T> sin(const Vec4<T>& v)
    {
        return Vec4<T>(sin(v.x), sin(v.y), sin(v.z), sin(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> cos(const Vec4<T>& v)
    {
        return Vec4<T>(cos(v.x), cos(v.y), cos(v.z), cos(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> tan(const Vec4<T>& v)
    {
        return Vec4<T>(tan(v.x), tan(v.y), tan(v.z), tan(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> asin(const Vec4<T>& v)
    {
        return Vec4<T>(asin(v.x), asin(v.y), asin(v.z), asin(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> acos(const Vec4<T>& v)
    {
        return Vec4<T>(acos(v.x), acos(v.y), acos(v.z), acos(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> atan(const Vec4<T>& v)
    {
        return Vec4<T>(atan(v.x), atan(v.y), atan(v.z), atan(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> sinh(const Vec4<T>& v)
    {
        return Vec4<T>(sinh(v.x), sinh(v.y), sinh(v.z), sinh(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> cosh(const Vec4<T>& v)
    {
        return Vec4<T>(cosh(v.x), cosh(v.y), cosh(v.z), cosh(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> tanh(const Vec4<T>& v)
    {
        return Vec4<T>(tanh(v.x), tanh(v.y), tanh(v.z), tanh(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> asinh(const Vec4<T>& v)
    {
        return Vec4<T>(asinh(v.x), asinh(v.y), asinh(v.z), asinh(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> acosh(const Vec4<T>& v)
    {
        return Vec4<T>(acosh(v.x), acosh(v.y), acosh(v.z), acosh(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> atanh(const Vec4<T>& v)
    {
        return Vec4<T>(atanh(v.x), atanh(v.y), atanh(v.z), atanh(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> pow(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>(
            pow(v1.x, v2.x), pow(v1.y, v2.y), pow(v1.z, v2.z), pow(v1.w, v2.w)
        );
    }

    template<std::floating_point T>
    inline Vec4<T> pow(const Vec4<T>& v1, T v2)
    {
        return Vec4<T>(
            pow(v1.x, v2), pow(v1.y, v2), pow(v1.z, v2), pow(v1.w, v2)
        );
    }

    template<std::floating_point T>
    inline Vec4<T> exp(const Vec4<T>& v)
    {
        return Vec4<T>(exp(v.x), exp(v.y), exp(v.z), exp(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> log(const Vec4<T>& v)
    {
        return Vec4<T>(log(v.x), log(v.y), log(v.z), log(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> exp2(const Vec4<T>& v)
    {
        return Vec4<T>(exp2(v.x), exp2(v.y), exp2(v.z), exp2(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> log2(const Vec4<T>& v)
    {
        return Vec4<T>(log2(v.x), log2(v.y), log2(v.z), log2(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> sqrt(const Vec4<T>& v)
    {
        return Vec4<T>(sqrt(v.x), sqrt(v.y), sqrt(v.z), sqrt(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> inversesqrt(const Vec4<T>& v)
    {
        return Vec4<T>(
            inversesqrt(v.x),
            inversesqrt(v.y),
            inversesqrt(v.z),
            inversesqrt(v.w)
        );
    }

    template<typename T>
    inline Vec4<T> abs(const Vec4<T>& v)
    {
        return Vec4<T>(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
    }

    template<typename T>
    constexpr Vec4<T> sign(const Vec4<T>& v)
    {
        return Vec4<T>(sign(v.x), sign(v.y), sign(v.z), sign(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> floor(const Vec4<T>& v)
    {
        return Vec4<T>(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> ceil(const Vec4<T>& v)
    {
        return Vec4<T>(ceil(v.x), ceil(v.y), ceil(v.z), ceil(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> trunc(const Vec4<T>& v)
    {
        return Vec4<T>(trunc(v.x), trunc(v.y), trunc(v.z), trunc(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> fract(const Vec4<T>& v)
    {
        return Vec4<T>(fract(v.x), fract(v.y), fract(v.z), fract(v.w));
    }

    template<std::floating_point T>
    inline Vec4<T> mod(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>(
            mod(v1.x, v2.x), mod(v1.y, v2.y), mod(v1.z, v2.z), mod(v1.w, v2.w)
        );
    }

    template<std::floating_point T>
    inline Vec4<T> mod(const Vec4<T>& v1, T v2)
    {
        return Vec4<T>(
            mod(v1.x, v2), mod(v1.y, v2), mod(v1.z, v2), mod(v1.w, v2)
        );
    }

    template<std::floating_point T>
    inline Vec4<T> modf(const Vec4<T>& v, Vec4<T>& i)
    {
        return Vec4<T>(
            modf(v.x, i.x), modf(v.y, i.y), modf(v.z, i.z), modf(v.w, i.w)
        );
    }

    template<std::floating_point T>
    inline Vec4<T> wrap(const Vec4<T>& v, T start, T end)
    {
        return start + mod(v - start, end - start);
    }

    template<typename T>
    constexpr Vec4<T> min(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>(
            min(v1.x, v2.x),
            min(v1.y, v2.y),
            min(v1.z, v2.z),
            min(v1.w, v2.w)
        );
    }

    template<typename T>
    constexpr Vec4<T> min(const Vec4<T>& v1, T v2)
    {
        return Vec4<T>(
            min(v1.x, v2),
            min(v1.y, v2),
            min(v1.z, v2),
            min(v1.w, v2)
        );
    }

    template<typename T>
    constexpr Vec4<T> min(T v1, const Vec4<T>& v2)
    {
        return Vec4<T>(
            min(v1, v2.x),
            min(v1, v2.y),
            min(v1, v2.z),
            min(v1, v2.w)
        );
    }

    template<typename T>
    constexpr Vec4<T> max(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>(
            max(v1.x, v2.x),
            max(v1.y, v2.y),
            max(v1.z, v2.z),
            max(v1.w, v2.w)
        );
    }

    template<typename T>
    constexpr Vec4<T> max(const Vec4<T>& v1, T v2)
    {
        return Vec4<T>(
            max(v1.x, v2),
            max(v1.y, v2),
            max(v1.z, v2),
            max(v1.w, v2)
        );
    }

    template<typename T>
    constexpr Vec4<T> max(T v1, const Vec4<T>& v2)
    {
        return Vec4<T>(
            max(v1, v2.x),
            max(v1, v2.y),
            max(v1, v2.z),
            max(v1, v2.w)
        );
    }

    template<typename T>
    constexpr Vec4<T> clamp(const Vec4<T>& v, T min, T max)
    {
        return Vec4<T>(
            clamp(v.x, min, max),
            clamp(v.y, min, max),
            clamp(v.z, min, max),
            clamp(v.w, min, max)
        );
    }

    template<typename T>
    constexpr Vec4<T> clamp(
        const Vec4<T>& v,
        const Vec4<T>& min,
        const Vec4<T>& max
    )
    {
        return Vec4<T>(
            clamp(v.x, min.x, max.x),
            clamp(v.y, min.y, max.y),
            clamp(v.z, min.z, max.z),
            clamp(v.w, min.w, max.w)
        );
    }

    template<std::floating_point T>
    constexpr Vec4<T> clamp01(const Vec4<T>& v)
    {
        return Vec4<T>(
            clamp01(v.x), clamp01(v.y), clamp01(v.z), clamp01(v.w)
        );
    }

    template<std::floating_point T>
    constexpr Vec4<T> mix(
        const Vec4<T>& v1,
        const Vec4<T>& v2,
        T a
    )
    {
        return v1 + a * (v2 - v1);
    }

    template<std::floating_point T>
    constexpr Vec4<T> remap(
        const Vec4<T>& v,
        T a_start,
        T a_end,
        T b_start,
        T b_end
    )
    {
        return
            b_start + ((b_end - b_start) / (a_end - a_start)) * (v - a_start);
    }

    template<std::floating_point T>
    constexpr Vec4<T> remap_clamp(
        const Vec4<T>& v,
        T a_start,
        T a_end,
        T b_start,
        T b_end
    )
    {
        Vec4<T> t = clamp01((v - a_start) / (a_end - a_start));
        return b_start + t * (b_end - b_start);
    }

    template<std::floating_point T>
    constexpr Vec4<T> remap01(
        const Vec4<T>& v,
        T a_start,
        T a_end
    )
    {
        return clamp01((v - a_start) / (a_end - a_start));
    }

    template<typename T>
    constexpr Vec4<T> step(T edge, const Vec4<T>& v)
    {
        return Vec4<T>(
            step(edge, v.x), step(edge, v.y), step(edge, v.z), step(edge, v.w)
        );
    }

    template<std::floating_point T>
    constexpr Vec4<T> smoothstep(
        T edge0,
        T edge1,
        const Vec4<T>& v
    )
    {
        return Vec4<T>(
            smoothstep(edge0, edge1, v.x),
            smoothstep(edge0, edge1, v.y),
            smoothstep(edge0, edge1, v.z),
            smoothstep(edge0, edge1, v.w)
        );
    }

    template<typename T>
    constexpr T length_squared(const Vec4<T>& v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    template<std::floating_point T>
    inline T length(const Vec4<T>& v)
    {
        return sqrt(length_squared(v));
    }

    template<typename T>
    constexpr T distance_squared(
        const Vec4<T>& v1,
        const Vec4<T>& v2
    )
    {
        return length_squared(v1 - v2);
    }

    template<std::floating_point T>
    inline T distance(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return length(v1 - v2);
    }

    template<typename T>
    constexpr T dot(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
    }

    template<std::floating_point T>
    inline Vec4<T> normalize(const Vec4<T>& v)
    {
        return v / length(v);
    }

    template<std::floating_point T>
    constexpr Vec4<T> faceforward(
        const Vec4<T>& n,
        const Vec4<T>& i,
        const Vec4<T>& nref
    )
    {
        if (dot(nref, i) < (T)0)
            return n;
        return -n;
    }

    template<std::floating_point T>
    constexpr Vec4<T> reflect(const Vec4<T>& i, const Vec4<T>& n)
    {
        return i - (T)2 * dot(n, i) * n;
    }

    template<std::floating_point T>
    inline Vec4<T> refract(
        const Vec4<T>& i,
        const Vec4<T>& n,
        T eta
    )
    {
        T dp = dot(n, i);
        T k = (T)1 - eta * eta * ((T)1 - dp * dp);

        if (k < (T)0)
            return (T)0;

        return eta * i - (eta * dp + sqrt(k)) * n;
    }

    template<typename T>
    constexpr T min_component(const Vec4<T>& v)
    {
        return min(v.x, min(v.y, min(v.z, v.w)));
    }

    template<typename T>
    constexpr T max_component(const Vec4<T>& v)
    {
        return max(v.x, max(v.y, max(v.z, v.w)));
    }

    template<typename T>
    constexpr i32 min_component_index(const Vec4<T>& v)
    {
        return (v.x < v.y)
            ? ((v.x < v.z)
                ? (v.x < v.w ? 0 : 3)
                : (v.z < v.w ? 2 : 3))
            : ((v.y < v.z)
                ? (v.y < v.w ? 1 : 3)
                : (v.z < v.w ? 2 : 3));
    }

    template<typename T>
    constexpr i32 max_component_index(const Vec4<T>& v)
    {
        return (v.x > v.y)
            ? ((v.x > v.z)
                ? (v.x > v.w ? 0 : 3)
                : (v.z > v.w ? 2 : 3))
            : ((v.y > v.z)
                ? (v.y > v.w ? 1 : 3)
                : (v.z > v.w ? 2 : 3));
    }

    using Vec4f = Vec4<f32>;
    using Vec4d = Vec4<f64>;
    using Vec4i = Vec4<i32>;
    using Vec4u = Vec4<u32>;

    // MARK: quaternions

    template<std::floating_point T>
    class Quat
    {
    public:
        Vec4<T> v;

        constexpr Quat()
            : v(0, 0, 0, 1)
        {}

        constexpr Quat(const Vec4<T>& v)
            : v(v)
        {}

        template<std::floating_point U>
        constexpr operator Quat<U>() const
        {
            return Quat<U>((Vec4<U>)v);
        }

        Quat(const Mat<T, 3, 3>& m)
        {
            T mtrace = m(0, 0) + m(1, 1) + m(2, 2);
            if (mtrace > 0)
            {
                // compute w from matrix trace, then xyz
                // 4w^2 = m(0, 0) + m(1, 1) + m(2, 2) + m(3, 3)
                // (but m(3, 3) == 1)
                T s = sqrt(mtrace + (T)1);
                v.w = s * (T).5;
                s = (T).5 / s;
                v.x = (m(2, 1) - m(1, 2)) * s;
                v.y = (m(0, 2) - m(2, 0)) * s;
                v.z = (m(1, 0) - m(0, 1)) * s;
            }
            else
            {
                // compute largest of $x$, $y$, or $z$, then remaining
                // components
                const i32 nxt[3] = { 1, 2, 0 };
                T q[3];
                i32 i = 0;
                if (m(1, 1) > m(0, 0))
                {
                    i = 1;
                }
                if (m(2, 2) > m(i, i))
                {
                    i = 2;
                }
                i32 j = nxt[i];
                i32 k = nxt[j];
                T s = sqrt((m(i, i) - (m(j, j) + m(k, k))) + (T)1);
                q[i] = s * (T).5;
                if (s != (T)0)
                {
                    s = (T).5 / s;
                }
                v.w = (m(k, j) - m(j, k)) * s;
                q[j] = (m(j, i) + m(i, j)) * s;
                q[k] = (m(k, i) + m(i, k)) * s;
                v.x = q[0];
                v.y = q[1];
                v.z = q[2];
            }
        }

        Quat(const Mat<T, 4, 4>& m)
            : Quat(m.sub<3>())
        {}

        std::string to_string() const
        {
            return v.to_string();
        }

        friend std::ostream& operator<<(
            std::ostream& os,
            const Quat& q
            )
        {
            os << q.to_string();
            return os;
        }

        constexpr Quat operator+(const Quat& q) const
        {
            return Quat(v + q.v);
        }

        constexpr Quat& operator+=(const Quat& q)
        {
            v += q.v;
            return *this;
        }

        constexpr Quat operator-(const Quat& q) const
        {
            return Quat(v - q.v);
        }

        constexpr Quat& operator-=(const Quat& q)
        {
            v -= q.v;
            return *this;
        }

        constexpr Quat operator*(T s) const
        {
            return Quat(v * s);
        }

        constexpr Quat& operator*=(T s)
        {
            v *= s;
            return *this;
        }

        constexpr Quat operator/(T s) const
        {
            return Quat(v / s);
        }

        constexpr Quat& operator/=(T s)
        {
            v /= s;
            return *this;
        }

        constexpr Quat operator-() const
        {
            return Quat(-v);
        }

        // generate a 3D homogeneous transformation matrix based on this
        // quaternion (left-handed)
        constexpr Mat<T, 4, 4> to_transform() const
        {
            T xx = v.x * v.x, yy = v.y * v.y, zz = v.z * v.z;
            T xy = v.x * v.y, xz = v.x * v.z, yz = v.y * v.z;
            T wx = v.x * v.w, wy = v.y * v.w, wz = v.z * v.w;

            Mat<T, 4, 4> r;
            r(0, 0) = 1 - 2 * (yy + zz);
            r(0, 1) = 2 * (xy + wz);
            r(0, 2) = 2 * (xz - wy);
            r(1, 0) = 2 * (xy - wz);
            r(1, 1) = 1 - 2 * (xx + zz);
            r(1, 2) = 2 * (yz + wx);
            r(2, 0) = 2 * (xz + wy);
            r(2, 1) = 2 * (yz - wx);
            r(2, 2) = 1 - 2 * (xx + yy);

            // transpose since we are left-handed
            return transpose(r);
        }

        constexpr bool operator==(const Quat& q) const
        {
            return v == q.v;
        }

        constexpr bool operator!=(const Quat& q) const
        {
            return v != q.v;
        }

    };

    template<std::floating_point T>
    constexpr Quat<T> operator*(T s, const Quat<T>& q)
    {
        return q * s;
    }

    template<std::floating_point T>
    constexpr T dot(
        const Quat<T>& q1,
        const Quat<T>& q2
    )
    {
        return dot(q1.v, q2.v);
    }

    template<std::floating_point T>
    constexpr Quat<T> normalize(const Quat<T>& q)
    {
        return Quat<T>(normalize(q.v));
    }

    // interpolate between two quaternions using spherical linear interpolation
    template<std::floating_point T>
    constexpr Quat<T> slerp(
        const Quat<T>& q1,
        const Quat<T>& q2,
        T t
    )
    {
        T cos_theta = dot(q1, q2);
        if (cos_theta > (T).9995)
        {
            return normalize(q1 + t * (q2 - q1));
        }
        else
        {
            T theta = acos(clamp(cos_theta, (T)(-1), (T)1));
            T thetap = theta * t;
            Quat qperp = normalize(q2 - q1 * cos_theta);
            return q1 * cos(thetap) + qperp * sin(thetap);
        }
    }

    using QuatF = Quat<f32>;
    using QuatD = Quat<f64>;

    // MARK: polar coordinates

    template<std::floating_point T>
    class Polar
    {
    public:
        T r;
        T theta;

        constexpr Polar()
            : r(0), theta(0)
        {}

        constexpr Polar(T r, T theta)
            : r(r), theta(theta)
        {}

        template<std::floating_point U>
        constexpr operator Polar<U>() const
        {
            return Polar<U>((U)r, (U)theta);
        }

        Polar(const Vec2<T>& cartesian)
        {
            r = length(cartesian);

            theta = atan(cartesian.y, cartesian.x);
            if (theta < 0)
                theta += tau<T>;
        }

        std::string to_string() const
        {
            return std::string("[r=") + to_string(r) + ", theta="
                + to_string(theta) + ']';
        }

        friend std::ostream& operator<<(std::ostream& os, const Polar& p)
        {
            os << p.to_string();
            return os;
        }

        constexpr bool operator==(const Polar& p) const
        {
            return r == p.r && theta == p.theta;
        }

        constexpr bool operator!=(const Polar& p) const
        {
            return r != p.r || theta != p.theta;
        }

        const Vec2<T> cartesian() const
        {
            return r * Vec2<T>(cos(theta), sin(theta));
        }

    };

    using PolarF = Polar<f32>;
    using PolarD = Polar<f64>;

    // MARK: spherical coordinates

    template<std::floating_point T>
    class Spherical
    {
    public:
        T r;
        T theta;
        T phi;

        constexpr Spherical()
            : r(0), theta(0), phi(0)
        {}

        constexpr Spherical(T r, T theta, T phi)
            : r(r), theta(theta), phi(phi)
        {}

        template<std::floating_point U>
        constexpr operator Spherical<U>() const
        {
            return Spherical<U>((U)r, (U)theta, (U)phi);
        }

        Spherical(const Vec3<T>& cartesian)
        {
            r = length(cartesian);
            theta = atan(r, cartesian.z);
            phi = atan(cartesian.y, cartesian.x);
        }

        std::string to_string() const
        {
            return std::string("[r=") + to_string(r)
                + ", theta=" + to_string(theta)
                + ", phi=" + to_string(phi)
                + ']';
        }

        friend std::ostream& operator<<(
            std::ostream& os,
            const Spherical& s
            )
        {
            os << s.to_string();
            return os;
        }

        constexpr bool operator==(const Spherical& s) const
        {
            return r == s.r && theta == s.theta && phi == s.phi;
        }

        constexpr bool operator!=(const Spherical& s) const
        {
            return r != s.r || theta != s.theta || phi != s.phi;
        }

        const Vec3<T> cartesian() const
        {
            const T sin_theta = sin(theta);
            return r * Vec3<T>(
                sin_theta * cos(phi),
                sin_theta * sin(phi),
                cos(theta)
            );
        }

    };

    using SphericalF = Spherical<f32>;
    using SphericalD = Spherical<f64>;

    // MARK: rays

    template<std::floating_point T>
    class Ray3
    {
    public:
        Vec3<T> o; // origin
        Vec3<T> d; // direction

        constexpr Ray3(const Vec3<T>& o, const Vec3<T>& d)
            : o(o), d(d)
        {}

        template<std::floating_point U>
        constexpr operator Ray3<U>() const
        {
            return Ray3<U>((Vec3<U>)o, (Vec3<U>)d);
        }

        // evaluate point along the ray
        constexpr Vec3<T> operator()(T t) const
        {
            return o + d * t;
        }

        std::string to_string() const
        {
            return std::string("[o=") + o.to_string() + ", d="
                + d.to_string() + ']';
        }

        friend std::ostream& operator<<(std::ostream& os, const Ray3& r)
        {
            os << r.to_string();
            return os;
        }

        constexpr bool operator==(const Ray3& r) const
        {
            return o == r.o && d == r.d;
        }

        constexpr bool operator!=(const Ray3& r) const
        {
            return o != r.o || d != r.d;
        }

    };

    using Ray3f = Ray3<f32>;
    using Ray3d = Ray3<f64>;

    // MARK: axis-aligned bounding boxes (AABB)

    template<typename T>
    class Aabb2
    {
    public:
        Vec2<T> pmin, pmax;

        constexpr Aabb2()
            : pmin(Vec2<T>(std::numeric_limits<T>::max())),
            pmax(Vec2<T>(std::numeric_limits<T>::lowest()))
        {}

        constexpr Aabb2(const Vec2<T>& p)
            : pmin(p), pmax(p)
        {}

        constexpr Aabb2(const Vec2<T>& p1, const Vec2<T>& p2)
            : pmin(min(p1, p2)), pmax(max(p1, p2))
        {}

        template<typename U>
        constexpr operator Aabb2<U>() const
        {
            return Aabb2<U>((Vec2<U>)pmin, (Vec2<U>)pmax);
        }

        std::string to_string() const
        {
            return std::string("[pmin=") + pmin.to_string()
                + ", pmax=" + pmax.to_string()
                + ']';
        }

        friend std::ostream& operator<<(
            std::ostream& os,
            const Aabb2<T>& b
            )
        {
            os << b.to_string();
            return os;
        }

        constexpr const Vec2<T>& operator[](i32 i) const
        {
            if (i == 0) return pmin;
            return pmax;
        }

        constexpr Vec2<T>& operator[](i32 i)
        {
            if (i == 0) return pmin;
            return pmax;
        }

        constexpr bool operator==(const Aabb2<T>& b) const
        {
            return pmin == b.pmin && pmax == b.pmax;
        }

        constexpr bool operator!=(const Aabb2<T>& b) const
        {
            return pmin != b.pmin || pmax != b.pmax;
        }

        // vector along the box diagonal from the minimum point to the maximum
        // point
        constexpr Vec2<T> diagonal() const
        {
            return pmax - pmin;
        }

        constexpr T area() const
        {
            Vec2<T> d = diagonal();
            return d.x * d.y;
        }

        // index of which of the axes is longest
        constexpr i32 max_extent() const
        {
            Vec2<T> d = diagonal();
            if (d.x > d.y)
                return 0;
            else
                return 1;
        }

        // linear interpolation between the corners of the box by the given
        // amount in each dimension
        constexpr Vec2<T> lerp(const Vec2<T>& t) const
        {
            return Vec2<T>(
                mix(pmin.x, pmax.x, t.x),
                mix(pmin.y, pmax.y, t.y)
            );
        }

        // the continuous position of a point relative to the corners of the
        // box, where a point at the minimum corner has offset (0, 0), a point
        // at the maximum corner has offset (1, 1), and so forth.
        constexpr Vec2<T> offset_of(const Vec2<T>& p) const
        {
            Vec2<T> o = p - pmin;
            if (pmax.x > pmin.x) o.x /= pmax.x - pmin.x;
            if (pmax.y > pmin.y) o.y /= pmax.y - pmin.y;
            return o;
        }

    };

    template<typename T>
    constexpr Aabb2<T> union_(
        const Aabb2<T>& b,
        const Vec2<T>& p
    )
    {
        return Aabb2<T>(min(b.pmin, p), max(b.pmax, p));
    }

    template<typename T>
    constexpr Aabb2<T> union_(
        const Aabb2<T>& b1,
        const Aabb2<T>& b2
    )
    {
        return Aabb2<T>(min(b1.pmin, b2.pmin), max(b1.pmax, b2.pmax));
    }

    template<typename T>
    constexpr Aabb2<T> intersect(
        const Aabb2<T>& b1,
        const Aabb2<T>& b2
    )
    {
        return Aabb2<T>(max(b1.pmin, b2.pmin), min(b1.pmax, b2.pmax));
    }

    template<typename T>
    constexpr bool overlaps(
        const Aabb2<T>& b1,
        const Aabb2<T>& b2
    )
    {
        return b1.pmax.x >= b2.pmin.x && b1.pmin.x <= b2.pmax.x
            && b1.pmax.y >= b2.pmin.y && b1.pmin.y <= b2.pmax.y;
    }

    template<typename T>
    constexpr bool inside(
        const Vec2<T>& p,
        const Aabb2<T>& b
    )
    {
        return p.x >= b.pmin.x && p.x <= b.pmax.x
            && p.y >= b.pmin.y && p.y <= b.pmax.y;
    }

    // the inside_exclusive() variant of inside() doesn't consider points on the
    // upper boundary to be inside the bounds. it is mostly useful with
    // integer-typed bounds.
    template<typename T>
    constexpr bool inside_exclusive(
        const Vec2<T>& p,
        const Aabb2<T>& b
    )
    {
        return p.x >= b.pmin.x && p.x < b.pmax.x
            && p.y >= b.pmin.y && p.y < b.pmax.y;
    }

    // pad the bounding box by a constant factor in all dimensions
    template<typename T, typename U>
    constexpr Aabb2<T> expand(const Aabb2<T>& b, U delta)
    {
        return Aabb2<T>(
            b.pmin - Vec2<T>((T)delta), b.pmax + Vec2<T>((T)delta)
        );
    }

    using Aabb2f = Aabb2<f32>;
    using Aabb2d = Aabb2<f64>;
    using Aabb2i = Aabb2<int>;

    template<typename T>
    class Aabb3
    {
    public:
        Vec3<T> pmin, pmax;

        constexpr Aabb3()
            : pmin(Vec3<T>(std::numeric_limits<T>::max())),
            pmax(Vec3<T>(std::numeric_limits<T>::lowest()))
        {}

        constexpr Aabb3(const Vec3<T>& p)
            : pmin(p), pmax(p)
        {}

        constexpr Aabb3(const Vec3<T>& p1, const Vec3<T>& p2)
            : pmin(min(p1, p2)), pmax(max(p1, p2))
        {}

        template<typename U>
        constexpr operator Aabb3<U>() const
        {
            return Aabb3<U>((Vec3<U>)pmin, (Vec3<U>)pmax);
        }

        std::string to_string() const
        {
            return std::string("[pmin=") + pmin.to_string()
                + ", pmax=" + pmax.to_string()
                + ']';
        }

        friend std::ostream& operator<<(
            std::ostream& os,
            const Aabb3<T>& b
            )
        {
            os << b.to_string();
            return os;
        }

        constexpr const Vec3<T>& operator[](i32 i) const
        {
            if (i == 0) return pmin;
            return pmax;
        }

        constexpr Vec3<T>& operator[](i32 i)
        {
            if (i == 0) return pmin;
            return pmax;
        }

        constexpr bool operator==(const Aabb3<T>& b) const
        {
            return pmin == b.pmin && pmax == b.pmax;
        }

        constexpr bool operator!=(const Aabb3<T>& b) const
        {
            return pmin != b.pmin || pmax != b.pmax;
        }

        // coordinates of one of the eight corners of the bounding box
        constexpr Vec3<T> corner(i32 i) const
        {
            return Vec3<T>(
                (*this)[(i & 1)].x,
                (*this)[(i & 2) ? 1 : 0].y,
                (*this)[(i & 4) ? 1 : 0].z
            );
        }

        // vector along the box diagonal from the minimum point to the maximum
        // point
        constexpr Vec3<T> diagonal() const
        {
            return pmax - pmin;
        }

        // surface area of the six faces of the box
        constexpr T surface_area() const
        {
            Vec3<T> d = diagonal();
            return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
        }

        constexpr T volume() const
        {
            Vec3<T> d = diagonal();
            return d.x * d.y * d.z;
        }

        // index of which of the axes is longest
        constexpr i32 max_extent() const
        {
            Vec3<T> d = diagonal();
            if (d.x > d.y && d.x > d.z)
                return 0;
            else if (d.y > d.z)
                return 1;
            else
                return 2;
        }

        // linear interpolation between the corners of the box by the given
        // amount in each dimension
        constexpr Vec3<T> lerp(const Vec3<T>& t) const
        {
            return Vec3<T>(
                mix(pmin.x, pmax.x, t.x),
                mix(pmin.y, pmax.y, t.y),
                mix(pmin.z, pmax.z, t.z)
            );
        }

        // the continuous position of a point relative to the corners of the
        // box, where a point at the minimum corner has offset (0, 0, 0), a
        // point at the maximum corner has offset (1, 1, 1), and so forth.
        constexpr Vec3<T> offset_of(const Vec3<T>& p) const
        {
            Vec3<T> o = p - pmin;
            if (pmax.x > pmin.x) o.x /= pmax.x - pmin.x;
            if (pmax.y > pmin.y) o.y /= pmax.y - pmin.y;
            if (pmax.z > pmin.z) o.z /= pmax.z - pmin.z;
            return o;
        }

    };

    template<typename T>
    constexpr Aabb3<T> union_(
        const Aabb3<T>& b,
        const Vec3<T>& p
    )
    {
        return Aabb3<T>(min(b.pmin, p), max(b.pmax, p));
    }

    template<typename T>
    constexpr Aabb3<T> union_(
        const Aabb3<T>& b1,
        const Aabb3<T>& b2
    )
    {
        return Aabb3<T>(min(b1.pmin, b2.pmin), max(b1.pmax, b2.pmax));
    }

    template<typename T>
    constexpr Aabb3<T> intersect(
        const Aabb3<T>& b1,
        const Aabb3<T>& b2
    )
    {
        return Aabb3<T>(max(b1.pmin, b2.pmin), min(b1.pmax, b2.pmax));
    }

    template<typename T>
    constexpr bool overlaps(
        const Aabb3<T>& b1,
        const Aabb3<T>& b2
    )
    {
        return b1.pmax.x >= b2.pmin.x && b1.pmin.x <= b2.pmax.x
            && b1.pmax.y >= b2.pmin.y && b1.pmin.y <= b2.pmax.y
            && b1.pmax.z >= b2.pmin.z && b1.pmin.z <= b2.pmax.z;
    }

    template<typename T>
    constexpr bool inside(
        const Vec3<T>& p,
        const Aabb3<T>& b
    )
    {
        return p.x >= b.pmin.x && p.x <= b.pmax.x
            && p.y >= b.pmin.y && p.y <= b.pmax.y
            && p.z >= b.pmin.z && p.z <= b.pmax.z;
    }

    // the inside_exclusive() variant of inside() doesn't consider points on the
    // upper boundary to be inside the bounds. it is mostly useful with
    // integer-typed bounds.
    template<typename T>
    constexpr bool inside_exclusive(
        const Vec3<T>& p,
        const Aabb3<T>& b
    )
    {
        return p.x >= b.pmin.x && p.x < b.pmax.x
            && p.y >= b.pmin.y && p.y < b.pmax.y
            && p.z >= b.pmin.z && p.z < b.pmax.z;
    }

    // pad the bounding box by a constant factor in all dimensions
    template<typename T, typename U>
    constexpr Aabb3<T> expand(const Aabb3<T>& b, U delta)
    {
        return Aabb3<T>(
            b.pmin - Vec3<T>((T)delta),
            b.pmax + Vec3<T>((T)delta)
        );
    }

    using Aabb3f = Aabb3<f32>;
    using Aabb3d = Aabb3<f64>;
    using Aabb3i = Aabb3<int>;

    // MARK: circle

    template<std::floating_point T>
    class Circle
    {
    public:
        Vec2<T> center;
        T radius;

        constexpr Circle()
            : center(Vec2<T>(0)), radius(1)
        {}

        constexpr Circle(const Vec2<T>& center, T radius)
            : center(center), radius(radius)
        {}

        template<std::floating_point U>
        constexpr operator Circle<U>() const
        {
            return Circle<U>((Vec2<U>)center, (U)radius);
        }

        // construct a circle that bounds a given bounding box
        Circle(const Aabb2<T>& b)
            : center((b.pmin + b.pmax) * .5f),
            radius(inside(center, b) ? distance(center, b.pmax) : 0)
        {}

        std::string to_string() const
        {
            return std::string("[center=") + center.to_string()
                + ", radius=" + to_string(radius)
                + ']';
        }

        friend std::ostream& operator<<(
            std::ostream& os,
            const Circle& c
            )
        {
            os << c.to_string();
            return os;
        }

        constexpr bool operator==(const Circle& c) const
        {
            return center == c.center && radius == c.radius;
        }

        constexpr bool operator!=(const Circle& c) const
        {
            return center != c.center || radius != c.radius;
        }

        constexpr Aabb2<T> bounds() const
        {
            return Aabb2<T>(center - radius, center + radius);
        }

        // a point on the circle at a given angle
        Vec2<T> at(T angle) const
        {
            return center + radius * unit_at(angle);
        }

        // a point on the unit circle at a given angle
        static Vec2<T> unit_at(T angle)
        {
            return Vec2<T>(cos(angle), sin(angle));
        }

    };

    template<std::floating_point T>
    inline bool inside(const Vec2<T>& p, const Circle<T>& c)
    {
        return distance_squared(p, c.center) <= squared(c.radius);
    }

    template<std::floating_point T>
    inline bool overlaps(const Circle<T>& c1, const Circle<T>& c2)
    {
        return
            distance_squared(c1.center, c2.center)
            <= squared(c1.radius + c2.radius);
    }

    template<std::floating_point T>
    inline bool overlaps(const Circle<T>& c, const Aabb2<T>& b)
    {
        return inside(b.pmin, c)
            || inside(vec2(b.pmax.x, b.pmin.y), c)
            || inside(vec2(b.pmin.x, b.pmax.y), c)
            || inside(b.pmax, c);
    }

    template<std::floating_point T>
    inline bool overlaps(const Aabb2<T>& b, const Circle<T>& c)
    {
        return overlaps(c, b);
    }

    using CircleF = Circle<f32>;
    using CircleD = Circle<f64>;

    // MARK: sphere

    template<std::floating_point T>
    class Sphere
    {
    public:
        Vec3<T> center;
        T radius;

        constexpr Sphere()
            : center(Vec3<T>(0)), radius(1)
        {}

        constexpr Sphere(const Vec3<T>& center, T radius)
            : center(center), radius(radius)
        {}

        template<std::floating_point U>
        constexpr operator Sphere<U>() const
        {
            return Sphere<U>((Vec3<U>)center, (U)radius);
        }

        // construct a sphere that bounds a given bounding box
        Sphere(const Aabb3<T>& b)
            : center((b.pmin + b.pmax) * .5f),
            radius(inside(center, b) ? distance(center, b.pmax) : 0)
        {}

        std::string to_string() const
        {
            return std::string("[center=") + center.to_string()
                + ", radius=" + to_string(radius)
                + ']';
        }

        friend std::ostream& operator<<(
            std::ostream& os,
            const Sphere& s
            )
        {
            os << s.to_string();
            return os;
        }

        constexpr bool operator==(const Sphere& s) const
        {
            return center == s.center && radius == s.radius;
        }

        constexpr bool operator!=(const Sphere& s) const
        {
            return center != s.center || radius != s.radius;
        }

        constexpr Aabb3<T> bounds() const
        {
            return Aabb3<T>(center - radius, center + radius);
        }

        // a point on the sphere at given theta and phi angles
        Vec3<T> at(T theta, T phi) const
        {
            return center + radius * unit_at(theta, phi);
        }

        // a point on the unit sphere at given theta and phi angles
        static Vec3<T> unit_at(T theta, T phi)
        {
            const T sin_theta = sin(theta);
            return Vec3<T>(
                sin_theta * cos(phi),
                sin_theta * sin(phi),
                cos(theta)
            );
        }

    };

    template<std::floating_point T>
    inline bool inside(const Vec3<T>& p, const Sphere<T>& s)
    {
        return distance_squared(p, s.center) <= squared(s.radius);
    }

    template<std::floating_point T>
    inline bool overlaps(const Sphere<T>& s1, const Sphere<T>& s2)
    {
        return
            distance_squared(s1.center, s2.center)
            <= squared(s1.radius + s2.radius);
    }

    template<std::floating_point T>
    inline bool overlaps(const Sphere<T>& s, const Aabb3<T>& b)
    {
        return inside(b.pmin, s)
            || inside(Vec3<T>(b.pmax.x, b.pmin.y, b.pmin.z), s)
            || inside(Vec3<T>(b.pmin.x, b.pmax.y, b.pmin.z), s)
            || inside(Vec3<T>(b.pmax.x, b.pmax.y, b.pmin.z), s)
            || inside(Vec3<T>(b.pmin.x, b.pmin.y, b.pmax.z), s)
            || inside(Vec3<T>(b.pmax.x, b.pmin.y, b.pmax.z), s)
            || inside(Vec3<T>(b.pmin.x, b.pmax.y, b.pmax.z), s)
            || inside(b.pmax, s);
    }

    template<std::floating_point T>
    inline bool overlaps(const Aabb3<T>& b, const Sphere<T>& s)
    {
        return overlaps(s, b);
    }

    using SphereF = Sphere<f32>;
    using SphereD = Sphere<f64>;

}

namespace axium::transform
{

    // MARK: linear transformations

    // 2D translation matrix (homogeneous)
    template<std::floating_point T>
    Mat<T, 3, 3> translate_2d_h(
        const Vec2<T>& delta,
        Mat<T, 3, 3>* out_minv = nullptr
    )
    {
        if (out_minv)
        {
            *out_minv = Mat<T, 3, 3>({
                1, 0, -delta.x,
                0, 1, -delta.y,
                0, 0, 1 }
                );
        }

        return Mat<T, 3, 3>({
            1, 0, delta.x,
            0, 1, delta.y,
            0, 0, 1 }
            );
    }

    // 3D translation matrix (homogeneous)
    template<std::floating_point T>
    Mat<T, 4, 4> translate_3d_h(
        const Vec3<T>& delta,
        Mat<T, 4, 4>* out_minv = nullptr
    )
    {
        if (out_minv)
        {
            *out_minv = Mat<T, 4, 4>({
                1, 0, 0, -delta.x,
                0, 1, 0, -delta.y,
                0, 0, 1, -delta.z,
                0, 0, 0, 1 }
                );
        }

        return Mat<T, 4, 4>({
            1, 0, 0, delta.x,
            0, 1, 0, delta.y,
            0, 0, 1, delta.z,
            0, 0, 0, 1 }
            );
    }

    // 2D scaling matrix
    template<std::floating_point T>
    Mat<T, 2, 2> scale_2d(
        const Vec2<T>& fac,
        Mat<T, 2, 2>* out_minv = nullptr
    )
    {
        if (out_minv)
        {
            *out_minv = Mat<T, 2, 2>({
                1 / fac.x, 0,
                0, 1 / fac.y }
                );
        }

        return Mat<T, 2, 2>({
            fac.x, 0,
            0, fac.y }
            );
    }

    // 2D scaling matrix (homogeneous)
    template<std::floating_point T>
    Mat<T, 3, 3> scale_2d_h(
        const Vec2<T>& fac,
        Mat<T, 3, 3>* out_minv = nullptr
    )
    {
        if (out_minv)
        {
            *out_minv = Mat<T, 3, 3>({
                1 / fac.x, 0, 0,
                0, 1 / fac.y, 0,
                0, 0, 1 }
                );
        }

        return Mat<T, 3, 3>({
            fac.x, 0, 0,
            0, fac.y, 0,
            0, 0, 1 }
            );
    }

    // 3D scaling matrix
    template<std::floating_point T>
    Mat<T, 3, 3> scale_3d(
        const Vec3<T>& fac,
        Mat<T, 3, 3>* out_minv = nullptr
    )
    {
        if (out_minv)
        {
            *out_minv = Mat<T, 3, 3>({
                1 / fac.x, 0, 0,
                0, 1 / fac.y, 0,
                0, 0, 1 / fac.z }
                );
        }

        return Mat<T, 3, 3>({
            fac.x, 0, 0,
            0, fac.y, 0,
            0, 0, fac.z }
            );
    }

    // 3D scaling matrix (homogeneous)
    template<std::floating_point T>
    Mat<T, 4, 4> scale_3d_h(
        const Vec3<T>& fac,
        Mat<T, 4, 4>* out_minv = nullptr
    )
    {
        if (out_minv)
        {
            *out_minv = Mat<T, 4, 4>({
                1 / fac.x, 0, 0, 0,
                0, 1 / fac.y, 0, 0,
                0, 0, 1 / fac.z, 0,
                0, 0, 0, 1 }
                );
        }

        return Mat<T, 4, 4>({
            fac.x, 0, 0, 0,
            0, fac.y, 0, 0,
            0, 0, fac.z, 0,
            0, 0, 0, 1 }
            );
    }

    // 2D rotation matrix
    template<std::floating_point T>
    Mat<T, 2, 2> rotate_2d(
        T angle,
        Mat<T, 2, 2>* out_minv = nullptr
    )
    {
        T s = sin(angle);
        T c = cos(angle);

        if (out_minv)
        {
            *out_minv = Mat<T, 2, 2>({
                c, s,
                -s, c }
                );
        }

        return Mat<T, 2, 2>({
            c, -s,
            s, c }
            );
    }

    // 2D rotation matrix (homogeneous)
    template<std::floating_point T>
    Mat<T, 3, 3> rotate_2d_h(
        T angle,
        Mat<T, 3, 3>* out_minv = nullptr
    )
    {
        T s = sin(angle);
        T c = cos(angle);

        if (out_minv)
        {
            *out_minv = Mat<T, 3, 3>({
                c, s, 0,
                -s, c, 0,
                0, 0, 1 }
                );
        }

        return Mat<T, 3, 3>({
            c, -s, 0,
            s, c, 0,
            0, 0, 1 }
            );
    }

    // 3D rotation matrix around the X axis (left-handed)
    template<std::floating_point T>
    Mat<T, 3, 3> rotate_3d_x(
        T angle,
        Mat<T, 3, 3>* out_minv = nullptr
    )
    {
        T s = sin(angle);
        T c = cos(angle);

        Mat<T, 3, 3> r({
            1, 0, 0,
            0, c, -s,
            0, s, c }
            );

        if (out_minv)
            *out_minv = transpose(r);

        return r;
    }

    // 3D rotation matrix around the X axis (left-handed) (homogeneous)
    template<std::floating_point T>
    Mat<T, 4, 4> rotate_3d_x_h(
        T angle,
        Mat<T, 4, 4>* out_minv = nullptr
    )
    {
        T s = sin(angle);
        T c = cos(angle);

        Mat<T, 4, 4> r({
            1, 0, 0, 0,
            0, c, -s, 0,
            0, s, c, 0,
            0, 0, 0, 1 }
            );

        if (out_minv)
            *out_minv = transpose(r);

        return r;
    }

    // 3D rotation matrix around the Y axis (left-handed)
    template<std::floating_point T>
    Mat<T, 3, 3> rotate_3d_y(
        T angle,
        Mat<T, 3, 3>* out_minv = nullptr
    )
    {
        T s = sin(angle);
        T c = cos(angle);

        Mat<T, 3, 3> r({
            c, 0, s,
            0, 1, 0,
            -s, 0, c }
            );

        if (out_minv)
            *out_minv = transpose(r);

        return r;
    }

    // 3D rotation matrix around the Y axis (left-handed) (homogeneous)
    template<std::floating_point T>
    Mat<T, 4, 4> rotate_3d_y_h(
        T angle,
        Mat<T, 4, 4>* out_minv = nullptr
    )
    {
        T s = sin(angle);
        T c = cos(angle);

        Mat<T, 4, 4> r({
            c, 0, s, 0,
            0, 1, 0, 0,
            -s, 0, c, 0,
            0, 0, 0, 1 }
            );

        if (out_minv)
            *out_minv = transpose(r);

        return r;
    }

    // 3D rotation matrix around the Z axis (left-handed)
    template<std::floating_point T>
    Mat<T, 3, 3> rotate_3d_z(
        T angle,
        Mat<T, 3, 3>* out_minv = nullptr
    )
    {
        T s = sin(angle);
        T c = cos(angle);

        Mat<T, 3, 3> r({
            c, -s, 0,
            s, c, 0,
            0, 0, 1 }
            );

        if (out_minv)
            *out_minv = transpose(r);

        return r;
    }

    // 3D rotation matrix around the Z axis (left-handed) (homogeneous)
    template<std::floating_point T>
    Mat<T, 4, 4> rotate_3d_z_h(
        T angle,
        Mat<T, 4, 4>* out_minv = nullptr
    )
    {
        T s = sin(angle);
        T c = cos(angle);

        Mat<T, 4, 4> r({
            c, -s, 0, 0,
            s, c, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1 }
            );

        if (out_minv)
            *out_minv = transpose(r);

        return r;
    }

    // 3D rotation matrix around an arbitrary axis (left-handed)
    template<std::floating_point T>
    Mat<T, 3, 3> rotate_3d(
        T angle,
        const Vec3<T>& axis,
        Mat<T, 3, 3>* out_minv = nullptr
    )
    {
        Vec3<T> a = normalize(axis);
        T s = sin(angle);
        T c = cos(angle);

        Mat<T, 3, 3> r;

        // compute rotation of first basis vector
        r(0, 0) = a.x * a.x + (1 - a.x * a.x) * c;
        r(0, 1) = a.x * a.y * (1 - c) - a.z * s;
        r(0, 2) = a.x * a.z * (1 - c) + a.y * s;

        // second basis vector
        r(1, 0) = a.x * a.y * (1 - c) + a.z * s;
        r(1, 1) = a.y * a.y + (1 - a.y * a.y) * c;
        r(1, 2) = a.y * a.z * (1 - c) - a.x * s;

        // third basis vector
        r(2, 0) = a.x * a.z * (1 - c) - a.y * s;
        r(2, 1) = a.y * a.z * (1 - c) + a.x * s;
        r(2, 2) = a.z * a.z + (1 - a.z * a.z) * c;

        if (out_minv)
            *out_minv = transpose(r);

        return r;
    }

    // 3D rotation matrix around an arbitrary axis (left-handed) (homogeneous)
    template<std::floating_point T>
    Mat<T, 4, 4> rotate_3d_h(
        T angle,
        const Vec3<T>& axis,
        Mat<T, 4, 4>* out_minv = nullptr
    )
    {
        Vec3<T> a = normalize(axis);
        T s = sin(angle);
        T c = cos(angle);

        Mat<T, 4, 4> r;

        // compute rotation of first basis vector
        r(0, 0) = a.x * a.x + (1 - a.x * a.x) * c;
        r(0, 1) = a.x * a.y * (1 - c) - a.z * s;
        r(0, 2) = a.x * a.z * (1 - c) + a.y * s;
        r(0, 3) = (T)0;

        // second basis vector
        r(1, 0) = a.x * a.y * (1 - c) + a.z * s;
        r(1, 1) = a.y * a.y + (1 - a.y * a.y) * c;
        r(1, 2) = a.y * a.z * (1 - c) - a.x * s;
        r(1, 3) = (T)0;

        // third basis vector
        r(2, 0) = a.x * a.z * (1 - c) - a.y * s;
        r(2, 1) = a.y * a.z * (1 - c) + a.x * s;
        r(2, 2) = a.z * a.z + (1 - a.z * a.z) * c;
        r(2, 3) = (T)0;

        if (out_minv)
            *out_minv = transpose(r);

        return r;
    }

    // 3D homogeneous transformation from a left-handed viewing coordinate
    // system where the camera is at the origin looking along the +z axis, where
    // the +y axis is along the up direction
    template<std::floating_point T>
    Mat<T, 4, 4> lookat_3d_h(
        const Vec3<T>& pos,
        const Vec3<T>& look,
        const Vec3<T>& up,
        Mat<T, 4, 4>* out_minv = nullptr
    )
    {
        Mat<T, 4, 4> cam_to_world;

        // initialize fourth column of viewing matrix
        cam_to_world(0, 3) = pos.x;
        cam_to_world(1, 3) = pos.y;
        cam_to_world(2, 3) = pos.z;
        cam_to_world(3, 3) = (T)1;

        // initialize first three columns of viewing matrix
        Vec3<T> dir = normalize(look - pos);
        Vec3<T> right = normalize(cross(normalize(up), dir));
        Vec3<T> new_up = cross(dir, right);
        cam_to_world(0, 0) = right.x;
        cam_to_world(1, 0) = right.y;
        cam_to_world(2, 0) = right.z;
        cam_to_world(3, 0) = (T)0;
        cam_to_world(0, 1) = new_up.x;
        cam_to_world(1, 1) = new_up.y;
        cam_to_world(2, 1) = new_up.z;
        cam_to_world(3, 1) = (T)0;
        cam_to_world(0, 2) = dir.x;
        cam_to_world(1, 2) = dir.y;
        cam_to_world(2, 2) = dir.z;
        cam_to_world(3, 2) = (T)0;

        if (out_minv)
            *out_minv = cam_to_world;

        return inverse(cam_to_world);
    }

    // transform a 2D point using a 2x2 matrix
    template<std::floating_point T>
    constexpr Vec2<T> apply_point_2d(
        const Mat<T, 2, 2>& m,
        const Vec2<T>& p
    )
    {
        return Vec2<T>(m * Mat<T, 2, 1>(p));
    }

    // transform a 2D point using a 3x3 homogeneous matrix
    template<std::floating_point T>
    constexpr Vec2<T> apply_point_2d_h(
        const Mat<T, 3, 3>& m,
        const Vec2<T>& p
    )
    {
        Vec3<T> r(m * Mat<T, 3, 1>(Vec3<T>(p, (T)1)));
        if (r.z == 0)
            return r.permute(0, 1);
        else
            return r.permute(0, 1) / r.z;
    }

    // transform a 3D point using a 3x3 matrix
    template<std::floating_point T>
    constexpr Vec3<T> apply_point_3d(
        const Mat<T, 3, 3>& m,
        const Vec3<T>& p
    )
    {
        return Vec3<T>(m * Mat<T, 3, 1>(p));
    }

    // transform a 3D point using a 4x4 homogeneous matrix
    template<std::floating_point T>
    constexpr Vec3<T> apply_point_3d_h(
        const Mat<T, 4, 4>& m,
        const Vec3<T>& p
    )
    {
        Vec4<T> r(m * Mat<T, 4, 1>(Vec4<T>(p, (T)1)));
        if (r.w == 0)
            return r.permute(0, 1, 2);
        else
            return r.permute(0, 1, 2) / r.w;
    }

    // transform a 2D vector using a 2x2 matrix
    template<std::floating_point T>
    constexpr Vec2<T> apply_vector_2d(
        const Mat<T, 2, 2>& m,
        const Vec2<T>& v
    )
    {
        return Vec2<T>(m * Mat<T, 2, 1>(v));
    }

    // transform a 2D vector using a 3x3 homogeneous matrix
    template<std::floating_point T>
    constexpr Vec2<T> apply_vector_2d_h(
        const Mat<T, 3, 3>& m,
        const Vec2<T>& v
    )
    {
        return Vec2<T>(m.sub<2>() * Mat<T, 2, 1>(v));
    }

    // transform a 3D vector using a 3x3 matrix
    template<std::floating_point T>
    constexpr Vec3<T> apply_vector_3d(
        const Mat<T, 3, 3>& m,
        const Vec3<T>& v
    )
    {
        return Vec3<T>(m * Mat<T, 3, 1>(v));
    }

    // transform a 3D vector using a 4x4 homogeneous matrix
    template<std::floating_point T>
    constexpr Vec3<T> apply_vector_3d_h(
        const Mat<T, 4, 4>& m,
        const Vec3<T>& v
    )
    {
        return Vec3<T>(m.sub<3>() * Mat<T, 3, 1>(v));
    }

    // transform a 2D normal vector using a 2x2 matrix
    // * you need to input the inverted version of your transformation matrix.
    //   use the inverse() function if you only have the original transformation
    //   matrix.
    template<std::floating_point T>
    constexpr Vec2<T> apply_normal_2d(
        const Mat<T, 2, 2>& minv,
        const Vec2<T>& n
    )
    {
        return Vec2<T>(transpose(minv) * Mat<T, 2, 1>(n));
    }

    // transform a 2D normal vector using a 3x3 homogeneous matrix
    // * you need to input the inverted version of your transformation matrix.
    //   use the inverse() function if you only have the original transformation
    //   matrix.
    template<std::floating_point T>
    constexpr Vec2<T> apply_normal_2d_h(
        const Mat<T, 3, 3>& minv,
        const Vec2<T>& n
    )
    {
        return Vec2<T>(transpose(minv.sub<2>()) * Mat<T, 2, 1>(n));
    }

    // transform a 3D normal vector using a 3x3 matrix
    // * you need to input the inverted version of your transformation matrix.
    //   use the inverse() function if you only have the original transformation
    //   matrix.
    template<std::floating_point T>
    constexpr Vec3<T> apply_normal_3d(
        const Mat<T, 3, 3>& minv,
        const Vec3<T>& n
    )
    {
        return Vec3<T>(transpose(minv) * Mat<T, 3, 1>(n));
    }

    // transform a 3D normal vector using a 4x4 homogeneous matrix
    // * you need to input the inverted version of your transformation matrix.
    //   use the inverse() function if you only have the original transformation
    //   matrix.
    template<std::floating_point T>
    constexpr Vec3<T> apply_normal_3d_h(
        const Mat<T, 4, 4>& minv,
        const Vec3<T>& n
    )
    {
        return Vec3<T>(transpose(minv.sub<3>()) * Mat<T, 3, 1>(n));
    }

    // transform a 3D ray using a 3x3 matrix
    template<std::floating_point T>
    constexpr Ray3<T> apply_ray_3d(
        const Mat<T, 3, 3>& m,
        Ray3<T> r
    )
    {
        r.o = apply_point_3d(m, r.o);
        r.d = apply_vector_3d(m, r.d);
        return r;
    }

    // transform a 3D ray using a 4x4 homogeneous matrix
    template<std::floating_point T>
    constexpr Ray3<T> apply_ray_3d_h(
        const Mat<T, 4, 4>& m,
        Ray3<T> r
    )
    {
        r.o = apply_point_3d_h(m, r.o);
        r.d = apply_vector_3d_h(m, r.d);
        return r;
    }

    // transform a 2D AABB using a 2x2 matrix
    template<std::floating_point T>
    constexpr Aabb2<T> apply_aabb_2d(
        const Mat<T, 2, 2>& m,
        const Aabb2<T>& b
    )
    {
        Aabb2<T> r(apply_point_2d(m, b.pmin));
        r = union_(r, apply_point_2d(m, Vec2<T>(b.pmin.x, b.pmax.y)));
        r = union_(r, apply_point_2d(m, Vec2<T>(b.pmax.x, b.pmin.y)));
        r = union_(r, apply_point_2d(m, b.pmax));
        return r;
    }

    // transform a 2D AABB using a 3x3 homogeneous matrix
    template<std::floating_point T>
    constexpr Aabb2<T> apply_aabb_2d_h(
        const Mat<T, 3, 3>& m,
        const Aabb2<T>& b
    )
    {
        Aabb2<T> r(apply_point_2d_h(m, b.pmin));
        r = union_(r, apply_point_2d_h(m, Vec2<T>(b.pmin.x, b.pmax.y)));
        r = union_(r, apply_point_2d_h(m, Vec2<T>(b.pmax.x, b.pmin.y)));
        r = union_(r, apply_point_2d_h(m, b.pmax));
        return r;
    }

    // transform a 3D AABB using a 3x3 matrix
    template<std::floating_point T>
    constexpr Aabb3<T> apply_aabb_3d(
        const Mat<T, 3, 3>& m,
        const Aabb3<T>& b
    )
    {
        Aabb3<T> r(apply_point_3d(m, b.pmin));
        r = union_(r, apply_point_3d(
            m, Vec3<T>(b.pmax.x, b.pmin.y, b.pmin.z)
        ));
        r = union_(r, apply_point_3d(
            m, Vec3<T>(b.pmin.x, b.pmax.y, b.pmin.z)
        ));
        r = union_(r, apply_point_3d(
            m, Vec3<T>(b.pmin.x, b.pmin.y, b.pmax.z)
        ));
        r = union_(r, apply_point_3d(
            m, Vec3<T>(b.pmin.x, b.pmax.y, b.pmax.z)
        ));
        r = union_(r, apply_point_3d(
            m, Vec3<T>(b.pmax.x, b.pmax.y, b.pmin.z)
        ));
        r = union_(r, apply_point_3d(
            m, Vec3<T>(b.pmax.x, b.pmin.y, b.pmax.z)
        ));
        r = union_(r, apply_point_3d(m, b.pmax));
        return r;
    }

    // transform a 3D AABB using a 4x4 homogeneous matrix
    template<std::floating_point T>
    constexpr Aabb3<T> apply_aabb_3d_h(
        const Mat<T, 4, 4>& m,
        const Aabb3<T>& b
    )
    {
        Aabb3<T> r(apply_point_3d_h(m, b.pmin));
        r = union_(r, apply_point_3d_h(
            m, Vec3<T>(b.pmax.x, b.pmin.y, b.pmin.z)
        ));
        r = union_(r, apply_point_3d_h(
            m, Vec3<T>(b.pmin.x, b.pmax.y, b.pmin.z)
        ));
        r = union_(r, apply_point_3d_h(
            m, Vec3<T>(b.pmin.x, b.pmin.y, b.pmax.z)
        ));
        r = union_(r, apply_point_3d_h(
            m, Vec3<T>(b.pmin.x, b.pmax.y, b.pmax.z)
        ));
        r = union_(r, apply_point_3d_h(
            m, Vec3<T>(b.pmax.x, b.pmax.y, b.pmin.z)
        ));
        r = union_(r, apply_point_3d_h(
            m, Vec3<T>(b.pmax.x, b.pmin.y, b.pmax.z)
        ));
        r = union_(r, apply_point_3d_h(m, b.pmax));
        return r;
    }

    // check if a 2D transformation matrix has a scaling term in it
    template<std::floating_point T>
    constexpr bool has_scale_2d(const Mat<T, 2, 2>& m)
    {
        T la2 = length_squared(apply_vector_2d(m, Vec2<T>(1, 0)));
        T lb2 = length_squared(apply_vector_2d(m, Vec2<T>(0, 1)));
        return
            la2 < (T).9999 || la2 >(T)1.0001 ||
            lb2 < (T).9999 || lb2 >(T)1.0001;
    }

    // check if a 2D homogeneous transformation matrix has a scaling term in it
    template<std::floating_point T>
    constexpr bool has_scale_2d_h(const Mat<T, 3, 3>& m)
    {
        T la2 = length_squared(apply_vector_2d_h(m, Vec2<T>(1, 0)));
        T lb2 = length_squared(apply_vector_2d_h(m, Vec2<T>(0, 1)));
        return
            la2 < (T).9999 || la2 >(T)1.0001 ||
            lb2 < (T).9999 || lb2 >(T)1.0001;
    }

    // check if a 3D transformation matrix has a scaling term in it
    template<std::floating_point T>
    constexpr bool has_scale_3d(const Mat<T, 3, 3>& m)
    {
        T la2 = length_squared(apply_vector_3d(m, Vec3<T>(1, 0, 0)));
        T lb2 = length_squared(apply_vector_3d(m, Vec3<T>(0, 1, 0)));
        T lc2 = length_squared(apply_vector_3d(m, Vec3<T>(0, 0, 1)));
        return
            la2 < (T).9999 || la2 >(T)1.0001 ||
            lb2 < (T).9999 || lb2 >(T)1.0001 ||
            lc2 < (T).9999 || lc2 >(T)1.0001;
    }

    // check if a 3D homogeneous transformation matrix has a scaling term in it
    template<std::floating_point T>
    constexpr bool has_scale_3d_h(const Mat<T, 4, 4>& m)
    {
        T la2 = length_squared(apply_vector_3d_h(m, Vec3<T>(1, 0, 0)));
        T lb2 = length_squared(apply_vector_3d_h(m, Vec3<T>(0, 1, 0)));
        T lc2 = length_squared(apply_vector_3d_h(m, Vec3<T>(0, 0, 1)));
        return
            la2 < (T).9999 || la2 >(T)1.0001 ||
            lb2 < (T).9999 || lb2 >(T)1.0001 ||
            lc2 < (T).9999 || lc2 >(T)1.0001;
    }

    // check if handedness is changed by a 3D transformation matrix
    template<std::floating_point T>
    constexpr bool swaps_handedness_3d(const Mat<T, 3, 3>& m)
    {
        return determinant(m) < (T)0;
    }

    // check if handedness is changed by a 3D homogeneous transformation matrix
    template<std::floating_point T>
    constexpr bool swaps_handedness_3d_h(const Mat<T, 4, 4>& m)
    {
        return determinant(m.sub<3>()) < (T)0;
    }

}
