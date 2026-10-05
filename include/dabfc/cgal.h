#ifndef DABFC_CGAL_H_
#define DABFC_CGAL_H_

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Simple_cartesian.h>
using Kernel = CGAL::Simple_cartesian<double>;
using Point = Kernel::Point_3;
using Vector = Kernel::Vector_3;
using Triangle = Kernel::Triangle_3;
using Primitive = CGAL::AABB_triangle_primitive_3<Kernel, std::vector<Triangle>::const_iterator>;
using AABB_tree = CGAL::AABB_tree<CGAL::AABB_traits_3<Kernel, Primitive>>;


#endif // DABFC_CGAL_H_