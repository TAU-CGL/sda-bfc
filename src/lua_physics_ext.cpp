// PhysicsEx: additions to LightEngine3's physics Lua binding, registered from main.cpp.
// Only Bullet's headers are used; the objects themselves are owned by the engine.
#include <le3/le3.h>
#include <le3/scripting/le3_script_bindings.h>
#include <btBulletDynamicsCommon.h>
using namespace le3;

// PhysicsEx.set_margin(physicsComponent, metres) -> bool
// Bullet inflates every convex collider by its collision margin (4 cm by default), so contacts get reported while
// the meshes are still centimetres apart. Returns false while the rigid body does not exist yet.
FBIND(PhysicsEx, set_margin) GET_UDATA(component, LE3PhysicsComponent) GET_NUMBER(margin)
    btRigidBody* body = (btRigidBody*)component->getRigidBody();
    if (body) body->getCollisionShape()->setMargin((float)margin);
    PUSH_BOOL(body != nullptr)
FEND()

LIB(PhysicsEx, set_margin)

void registerPhysicsEx() {
    lua_State* L = LE3GetScriptSystem().getLuaState();
    luaL_requiref(L, "PhysicsEx", le3::luaopen_PhysicsEx, 1);
    lua_pop(L, 1);
}
