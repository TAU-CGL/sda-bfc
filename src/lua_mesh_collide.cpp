// MeshCollide: exact triangle-mesh intersection tests between named, posed meshes, exposed to Lua (core in mesh_collide.h).
//   MeshCollide.add(name, datPath) -> triangles       append a binary STL (metres) to mesh `name`
//   MeshCollide.set_pose(name, px,py,pz, qw,qx,qy,qz)  world pose of the mesh (quaternion w,x,y,z as in the engine)
//   MeshCollide.ignore(nameA, nameB)                   never report this pair (adjacent links)
//   MeshCollide.pairs() -> {a1, b1, a2, b2, ...}       every pair of meshes whose triangles currently intersect
#include <le3/le3.h>
#include <le3/scripting/le3_script_bindings.h>
#include <map>
#include <set>
#include "mesh_collide.h"
using namespace le3;
using meshcollide::Mesh, meshcollide::Tri;

static std::map<std::string, Mesh> meshes;
static std::set<std::pair<std::string, std::string>> ignored;

FBIND(MeshCollide, add) GET_STRING(name) GET_STRING(path)
    LE3DatBuffer buf = LE3GetDatFileSystem().getFileContent(path);
    const char* d = buf.data.data(); uint32_t n = 0;
    if (buf.data.size() >= 84) memcpy(&n, d + 80, 4);
    if (buf.data.size() < 84 + (size_t)n * 50) n = 0;
    Mesh& m = meshes[name];
    for (uint32_t i = 0; i < n; i++) { Tri t; memcpy(&t, d + 84 + i * 50 + 12, 36); m.tris.push_back(t); }
    m.nodes.clear(); if (!m.tris.empty()) meshcollide::build(m, 0, (int)m.tris.size());
    PUSH_NUMBER(n)
FEND()

FBIND(MeshCollide, set_pose) GET_STRING(name) GET_NUMBER(px) GET_NUMBER(py) GET_NUMBER(pz) GET_NUMBER(qw) GET_NUMBER(qx) GET_NUMBER(qy) GET_NUMBER(qz)
    Mesh& m = meshes[name]; m.p = glm::vec3(px, py, pz); m.q = glm::normalize(glm::quat((float)qw, (float)qx, (float)qy, (float)qz));
FEND()

FBIND(MeshCollide, ignore) GET_STRING(a) GET_STRING(b)
    ignored.insert(std::minmax(a, b));
FEND()

FBIND(MeshCollide, pairs)
    std::vector<std::string> out;
    for (auto a = meshes.begin(); a != meshes.end(); ++a) for (auto b = std::next(a); b != meshes.end(); ++b)
        if (!ignored.count({a->first, b->first}) && meshcollide::collide(a->second, b->second)) { out.push_back(a->first); out.push_back(b->first); }
    PUSH_STRING_ARRAY(out)
FEND()

LIB(MeshCollide, add, set_pose, ignore, pairs)

void registerMeshCollide() {
    lua_State* L = LE3GetScriptSystem().getLuaState();
    luaL_requiref(L, "MeshCollide", le3::luaopen_MeshCollide, 1);
    lua_pop(L, 1);
}
