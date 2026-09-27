// Exact triangle-mesh intersection tests between posed meshes (no engine dependencies; Lua binding in lua_mesh_collide.cpp).
// Each mesh keeps its triangles in local space under an AABB tree. A query walks two trees with one mesh expressed in the
// other's frame and stops at the first intersecting triangle pair (separating-axis test with a 0.01 mm tolerance), so
// a reported contact means the surfaces overlap: no hulls, no margins.
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <vector>

namespace meshcollide {
    constexpr float EPS = 1e-5f; // metres of overlap needed before two triangles count as intersecting
    struct Tri { glm::vec3 a, b, c; };
    struct Node { glm::vec3 lo, hi; int l = -1, r = -1, begin = 0, end = 0; }; // l < 0: leaf over tris[begin, end)
    struct Mesh { std::vector<Tri> tris; std::vector<Node> nodes; glm::vec3 p{0.f}; glm::quat q{1.f, 0.f, 0.f, 0.f}; };
    struct Rel { glm::mat3 R, A; glm::vec3 t; }; // B-local -> A-local rotation, its absolute value, translation
    inline long visits = 0; // node pairs visited, for profiling

    inline int build(Mesh& m, int begin, int end) {
        Node n; n.lo = glm::vec3(1e30f); n.hi = -n.lo; n.begin = begin; n.end = end;
        for (int i = begin; i < end; i++) for (const glm::vec3& v : {m.tris[i].a, m.tris[i].b, m.tris[i].c}) { n.lo = glm::min(n.lo, v); n.hi = glm::max(n.hi, v); }
        int idx = (int)m.nodes.size(); m.nodes.push_back(n);
        if (end - begin > 4) {
            glm::vec3 e = n.hi - n.lo; int axis = e.x > e.y ? (e.x > e.z ? 0 : 2) : (e.y > e.z ? 1 : 2);
            int mid = (begin + end) / 2;
            std::nth_element(m.tris.begin() + begin, m.tris.begin() + mid, m.tris.begin() + end,
                [axis](const Tri& x, const Tri& y) { return x.a[axis] + x.b[axis] + x.c[axis] < y.a[axis] + y.b[axis] + y.c[axis]; });
            int l = build(m, begin, mid), r = build(m, mid, end);
            m.nodes[idx].l = l; m.nodes[idx].r = r;
        }
        return idx;
    }

    inline bool separated(glm::vec3 axis, const Tri& t0, const Tri& t1, float scale2) {
        if (glm::dot(axis, axis) < 1e-12f * scale2) return false; // parallel edges: no verdict from this axis
        axis = glm::normalize(axis);
        float a0 = glm::dot(axis, t0.a), b0 = glm::dot(axis, t0.b), c0 = glm::dot(axis, t0.c);
        float a1 = glm::dot(axis, t1.a), b1 = glm::dot(axis, t1.b), c1 = glm::dot(axis, t1.c);
        return std::max({a0, b0, c0}) < std::min({a1, b1, c1}) + EPS || std::max({a1, b1, c1}) < std::min({a0, b0, c0}) + EPS;
    }

    inline bool intersect(const Tri& t0, const Tri& t1) {
        glm::vec3 e0[3] = {t0.b - t0.a, t0.c - t0.b, t0.a - t0.c}, e1[3] = {t1.b - t1.a, t1.c - t1.b, t1.a - t1.c};
        glm::vec3 n0 = glm::cross(e0[0], e0[1]), n1 = glm::cross(e1[0], e1[1]);
        if (separated(n0, t0, t1, 0.f) || separated(n1, t0, t1, 0.f)) return false;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
            if (separated(glm::cross(e0[i], e1[j]), t0, t1, glm::dot(e0[i], e0[i]) * glm::dot(e1[j], e1[j]))) return false;
        for (int i = 0; i < 3; i++) // in-plane edge normals settle the coplanar case
            if (separated(glm::cross(n0, e0[i]), t0, t1, 0.f) || separated(glm::cross(n1, e1[i]), t0, t1, 0.f)) return false;
        return true;
    }

    inline bool overlap(const Node& a, const Node& b, const Rel& r) { // AABB of a vs. AABB of b brought into a's frame
        glm::vec3 c = r.R * (0.5f * (b.lo + b.hi)) + r.t, h = r.A * (0.5f * (b.hi - b.lo));
        return glm::all(glm::lessThanEqual(a.lo, c + h)) && glm::all(glm::lessThanEqual(c - h, a.hi));
    }

    inline bool collide(const Mesh& A, int ia, const Mesh& B, int ib, const Rel& r) {
        const Node &na = A.nodes[ia], &nb = B.nodes[ib];
        visits++;
        if (!overlap(na, nb, r)) return false;
        if (na.l < 0 && nb.l < 0) {
            for (int i = na.begin; i < na.end; i++) for (int j = nb.begin; j < nb.end; j++) {
                Tri t = {r.R * B.tris[j].a + r.t, r.R * B.tris[j].b + r.t, r.R * B.tris[j].c + r.t};
                if (intersect(A.tris[i], t)) return true;
            }
            return false;
        }
        glm::vec3 ea = na.hi - na.lo, eb = nb.hi - nb.lo;
        if (nb.l < 0 || (na.l >= 0 && ea.x * ea.y * ea.z > eb.x * eb.y * eb.z)) return collide(A, na.l, B, ib, r) || collide(A, na.r, B, ib, r);
        return collide(A, ia, B, nb.l, r) || collide(A, ia, B, nb.r, r);
    }

    inline Rel relative(const Mesh& A, const Mesh& B) {
        glm::quat qa = glm::conjugate(A.q);
        Rel r; r.R = glm::mat3_cast(qa * B.q); r.t = qa * (B.p - A.p);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.A[i][j] = std::abs(r.R[i][j]);
        return r;
    }

    inline bool collide(const Mesh& A, const Mesh& B) { return !A.nodes.empty() && !B.nodes.empty() && collide(A, 0, B, 0, relative(A, B)); }
}
