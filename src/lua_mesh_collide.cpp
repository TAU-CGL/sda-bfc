// MeshCollide: exact triangle-mesh contact tests between named, posed meshes (FCL: OBBRSS trees + triangle SAT), for Lua.
//   MeshCollide.add(name, datPath) -> triangles       append a binary STL (metres) to mesh `name`
//   MeshCollide.set_pose(name, px,py,pz, qw,qx,qy,qz)  world pose of the mesh (quaternion w,x,y,z as in the engine)
//   MeshCollide.ignore(nameA, nameB)                   never report this pair (adjacent links)
//   MeshCollide.pairs() -> {a1, b1, a2, b2, ...}       every pair of meshes whose surfaces touch or overlap (no margin)
#include <le3/le3.h>
#include <le3/scripting/le3_script_bindings.h>
#include <fcl/fcl.h>
#include <map>
#include <set>
using namespace le3;

struct Mesh { std::vector<fcl::Vector3d> v; std::vector<fcl::Triangle> t; fcl::Transform3d tf = fcl::Transform3d::Identity(); std::unique_ptr<fcl::BVHModel<fcl::OBBRSSd>> bvh; };
static std::map<std::string, Mesh> meshes;
static std::set<std::pair<std::string, std::string>> ignored;
static std::vector<std::string> hits;
static bool dirty = true; // a pose, mesh or ignore changed since the last pairs()

FBIND(MeshCollide, add) GET_STRING(name) GET_STRING(path)
    LE3DatBuffer buf = LE3GetDatFileSystem().getFileContent(path);
    const char* d = buf.data.data(); uint32_t n = 0;
    if (buf.data.size() >= 84) memcpy(&n, d + 80, 4);
    if (buf.data.size() < 84 + (size_t)n * 50) n = 0;
    Mesh& m = meshes[name]; m.bvh.reset(); dirty = true;
    for (uint32_t i = 0; i < n; i++) {
        float f[9]; memcpy(f, d + 84 + i * 50 + 12, 36);
        m.t.emplace_back(m.v.size(), m.v.size() + 1, m.v.size() + 2);
        for (int k = 0; k < 9; k += 3) m.v.emplace_back(f[k], f[k + 1], f[k + 2]);
    }
    PUSH_NUMBER(n)
FEND()

FBIND(MeshCollide, set_pose) GET_STRING(name) GET_NUMBER(px) GET_NUMBER(py) GET_NUMBER(pz) GET_NUMBER(qw) GET_NUMBER(qx) GET_NUMBER(qy) GET_NUMBER(qz)
    fcl::Transform3d tf(Eigen::Translation3d(px, py, pz) * fcl::Quaterniond(qw, qx, qy, qz).normalized());
    Mesh& m = meshes[name];
    if (tf.matrix() != m.tf.matrix()) { m.tf = tf; dirty = true; }
FEND()

FBIND(MeshCollide, ignore) GET_STRING(a) GET_STRING(b)
    ignored.insert(std::minmax(a, b)); dirty = true;
FEND()

FBIND(MeshCollide, pairs)
    if (dirty) {
        dirty = false; hits.clear();
        for (auto& [_, m] : meshes) if (!m.bvh && !m.t.empty()) { m.bvh = std::make_unique<fcl::BVHModel<fcl::OBBRSSd>>(); m.bvh->beginModel(); m.bvh->addSubModel(m.v, m.t); m.bvh->endModel(); }
        fcl::CollisionRequestd req; // num_max_contacts = 1: stop at the first touching triangle pair
        for (auto a = meshes.begin(); a != meshes.end(); ++a) for (auto b = std::next(a); b != meshes.end(); ++b) {
            fcl::CollisionResultd res;
            if (a->second.bvh && b->second.bvh && !ignored.count({a->first, b->first}) && fcl::collide(a->second.bvh.get(), a->second.tf, b->second.bvh.get(), b->second.tf, req, res)) { hits.push_back(a->first); hits.push_back(b->first); }
        }
    }
    PUSH_STRING_ARRAY(hits)
FEND()

LIB(MeshCollide, add, set_pose, ignore, pairs)

void registerMeshCollide() {
    lua_State* L = LE3GetScriptSystem().getLuaState();
    luaL_requiref(L, "MeshCollide", le3::luaopen_MeshCollide, 1);
    lua_pop(L, 1);
}
