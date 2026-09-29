-- UR5e arm with a Robotiq 2F-85 gripper, built from the ur_description and robotiq URDFs, driven by joint values in radians.
-- Add in-editor as a ScriptObject with Classname = "UR5e"; the object's transform is the robot's base frame (Y-up).
-- Meshes are loaded from the resources/ur5e folder (packed by the editor into ur5e.dat): visual/*.stl (one file per URDF
-- material, baked from the .dae files by tools/dae2stl.py) are what you see, collision/*.stl feed Bullet as convex hulls.
-- Everything created here is prefixed with __ENGINE__ so that it is never written into the saved scene or shared assets.
-- Collisions: getCollisions() lists contacts with other scene objects (including other robots), getSelfCollisions() lists
-- contacts between this robot's own links except adjacent ones and pairs whitelisted with allowCurrentSelfCollisions().
-- Link-to-link contacts (own links and other robots) are exact intersections of the visual meshes, computed by the
-- engine's LE3MeshCollision (FCL); contacts with anything else come from Bullet's convex hulls.
UR5e = LE3ScriptObject:new()
UR5e.ignore = {} -- scene object names whose contacts never count, e.g. UR5e.ignore.table = true
UR5e.margin = 0.003 -- Bullet collision margin per link (metres); the engine default of 0.04 reports contacts ~8 cm before the meshes touch
UR5e.linkOf = {} -- link object name -> {arm = UR5e instance, link = link name}, for all robots in the scene

-- Editor material used for each material of the visual meshes. If a name does not exist in the project, a built-in
-- __ENGINE__ copy with the URDF colors is created instead, so the robot looks right without any editor setup
UR5e.materials = {LinkGrey = "M_ur5e_silver", JointGrey = "M_ur5e_silver", Black = "M_ur5e_silver", URBlue = "M_ur5e_blue",
                  RobotiqBlack = "M_robotiq_black", RobotiqGrey = "M_ur5e_silver"}
local URDF_COLORS = {M_ur5e_silver = {0.82, 0.82, 0.82, 1}, M_ur5e_blue = {0.49, 0.68, 0.80, 1}, M_robotiq_black = {0.22, 0.22, 0.22, 1}}

local PI, X, ZERO = math.pi, {1, 0, 0}, {0, 0, 0}
-- Kinematic tree (ur5e.urdf, robotiq_arg2f_85_model.urdf). Per row: parent row, driving joint (nil = fixed) and multiplier,
-- joint origin xyz/rpy and axis (default Z), mesh file base name, mesh origin mxyz/mrpy, visual parts (URDF material names)
local LINKS = {
    {name = "base",     mesh = "base",     mrpy = {0, 0, PI}, parts = {"JointGrey", "Black"}},
    {name = "shoulder", mesh = "shoulder", parent = 1, joint = 1, xyz = {0, 0, 0.1625}, mrpy = {0, 0, PI}, parts = {"JointGrey", "Black", "URBlue"}},
    {name = "upperarm", mesh = "upperarm", parent = 2, joint = 2, rpy = {PI/2, 0, 0}, mxyz = {0, 0, 0.138}, mrpy = {PI/2, 0, -PI/2}, parts = {"JointGrey", "Black", "URBlue", "LinkGrey"}},
    {name = "forearm",  mesh = "forearm",  parent = 3, joint = 3, xyz = {-0.425, 0, 0}, mxyz = {0, 0, 0.007}, mrpy = {PI/2, 0, -PI/2}, parts = {"JointGrey", "Black", "URBlue", "LinkGrey"}},
    {name = "wrist1",   mesh = "wrist1",   parent = 4, joint = 4, xyz = {-0.3922, 0, 0.1333}, mxyz = {0, 0, -0.127}, mrpy = {PI/2, 0, 0}, parts = {"JointGrey", "Black", "URBlue"}},
    {name = "wrist2",   mesh = "wrist2",   parent = 5, joint = 5, xyz = {0, -0.0997, 0}, rpy = {PI/2, 0, 0}, mxyz = {0, 0, -0.0997}, parts = {"JointGrey", "Black", "URBlue"}},
    {name = "wrist3",   mesh = "wrist3",   parent = 6, joint = 6, xyz = {0, 0.0996, 0}, rpy = {PI/2, PI, PI}, mxyz = {0, 0, -0.0989}, mrpy = {PI/2, 0, 0}, parts = {"LinkGrey"}},
    {name = "tool0", parent = 7}, -- flange frame (coincides with wrist_3_link); the gripper mounts here
    -- Robotiq 2F-85: joint 7 is finger_joint (0 = open, 0.8 = closed); the other finger joints mimic it
    {name = "gripper",         mesh = "rq_base_link",     parent = 8, parts = {"RobotiqBlack", "RobotiqGrey"}},
    {name = "l_outer_knuckle", mesh = "rq_outer_knuckle", parent = 9,  joint = 7, xyz = {0, -0.0306011, 0.054904}, rpy = {0, 0, PI}, axis = X, parts = {"RobotiqBlack"}},
    {name = "l_outer_finger",  mesh = "rq_outer_finger",  parent = 10, xyz = {0, 0.0315, -0.0041}, parts = {"RobotiqBlack"}},
    {name = "l_inner_finger",  mesh = "rq_inner_finger",  parent = 11, joint = 7, mult = -1, xyz = {0, 0.0061, 0.0471}, axis = X, parts = {"RobotiqBlack", "RobotiqGrey"}},
    {name = "l_inner_knuckle", mesh = "rq_inner_knuckle", parent = 9,  joint = 7, xyz = {0, -0.0127, 0.06142}, rpy = {0, 0, PI}, axis = X, parts = {"RobotiqBlack"}},
    {name = "r_outer_knuckle", mesh = "rq_outer_knuckle", parent = 9,  joint = 7, xyz = {0, 0.0306011, 0.054904}, axis = X, parts = {"RobotiqBlack"}},
    {name = "r_outer_finger",  mesh = "rq_outer_finger",  parent = 14, xyz = {0, 0.0315, -0.0041}, parts = {"RobotiqBlack"}},
    {name = "r_inner_finger",  mesh = "rq_inner_finger",  parent = 15, joint = 7, mult = -1, xyz = {0, 0.0061, 0.0471}, axis = X, parts = {"RobotiqBlack", "RobotiqGrey"}},
    {name = "r_inner_knuckle", mesh = "rq_inner_knuckle", parent = 9,  joint = 7, xyz = {0, 0.0127, 0.06142}, axis = X, parts = {"RobotiqBlack"}},
}
local TOOL0 = 8

-- Quaternion {w,x,y,z} and pose {p={x,y,z}, q} helpers
local function qmul(a, b)
    return {a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4], a[1]*b[2] + a[2]*b[1] + a[3]*b[4] - a[4]*b[3],
            a[1]*b[3] - a[2]*b[4] + a[3]*b[1] + a[4]*b[2], a[1]*b[4] + a[2]*b[3] - a[3]*b[2] + a[4]*b[1]}
end
local function qaxis(angle, x, y, z) local s = math.sin(angle / 2); return {math.cos(angle / 2), x * s, y * s, z * s} end
local function qrpy(rpy) return qmul(qaxis(rpy[3], 0, 0, 1), qmul(qaxis(rpy[2], 0, 1, 0), qaxis(rpy[1], 1, 0, 0))) end
local function compose(A, B) -- pose A followed by pose B (B expressed in A's frame)
    local t = qmul(qmul(A.q, {0, B.p[1], B.p[2], B.p[3]}), {A.q[1], -A.q[2], -A.q[3], -A.q[4]})
    return {p = {A.p[1] + t[2], A.p[2] + t[3], A.p[3] + t[4]}, q = qmul(A.q, B.q)}
end

-- Forward kinematics: world pose of every row's mesh, and of every row's frame (joints[7] is the gripper)
function UR5e.fk(root, joints)
    local T, poses = {}, {}
    root = compose(root, {p = ZERO, q = qaxis(-PI / 2, 1, 0, 0)}) -- URDF is Z-up, LE3 is Y-up
    for i, L in ipairs(LINKS) do
        local a, q = L.axis or {0, 0, 1}, L.joint and joints[L.joint] * (L.mult or 1) or 0
        T[i] = compose(L.parent and T[L.parent] or root, {p = L.xyz or ZERO, q = qmul(qrpy(L.rpy or ZERO), qaxis(q, a[1], a[2], a[3]))})
        poses[i] = compose(T[i], {p = L.mxyz or ZERO, q = qrpy(L.mrpy or ZERO)})
    end
    return poses, T
end

local function resolveMaterial(name) -- the editor material if it exists, otherwise the built-in copy
    if LE3AssetManager.has_material(name) then return name end
    if not LE3AssetManager.has_material("__ENGINE__" .. name) then
        LE3Material.load({Name = "__ENGINE__" .. name, ShaderName = "__ENGINE__S_default", DiffuseColor = URDF_COLORS[name] or {0.8, 0.8, 0.8, 1}, SpecularIntensity = 0.3, Shininess = 32})
    end
    return "__ENGINE__" .. name
end

-- Registers the mesh file (once) and adds a static model using it; extra fields of tbl go to LE3StaticModel.load
function UR5e:addModel(name, file, collider, material, tbl)
    local mesh = "__ENGINE__SM_ur5e_" .. file
    if not LE3AssetManager.has_static_mesh(mesh) then LE3StaticMesh.load({Name = mesh, Path = "/ur5e/" .. file, ColliderType = collider}) end
    tbl.Name, tbl.MeshName, tbl.MaterialName = name, mesh, resolveMaterial(material)
    LE3StaticModel.load(self.scene, tbl)
    return LE3Scene.get_object(self.scene, name)
end

function UR5e:init()
    self.joints, self.links, self.linkNames, self.hits, self.collisions = {0, -PI/2, 0, -PI/2, 0, 0, 0}, {}, {}, {}, {}
    self.selfHits, self.selfCollisions, self.allowed, self.objName, self.lastFree = {}, {}, {}, {}, {table.unpack(self.joints)}
    for i, L in ipairs(LINKS) do
        if L.mesh then
            local name = "__ENGINE__" .. self.name .. "_" .. L.name
            -- Hidden collision mesh: a kinematic rigid body (mass > 0, not simulated) so that Bullet reports contacts with everything
            local obj = self:addModel(name, "collision/" .. L.mesh .. ".stl", "ConvexHull", "M_default", {IsRigidBody = true, Mass = 1, Hidden = true})
            for _, part in ipairs(L.parts) do -- visual parts ride along as children of the collision body
                self:addModel(name .. "_" .. part, "visual/" .. L.mesh .. "_" .. part .. ".stl", nil, UR5e.materials[part], {IsRigidBody = false})
                LE3Scene.reparent(self.scene, name .. "_" .. part, name)
                LE3MeshCollision.add(name, "/ur5e/visual/" .. L.mesh .. "_" .. part .. ".stl")
            end
            self.links[i] = {transform = LE3Object.get_transform(obj), physics = LE3Object.get_physics_component(obj), obj = name}
            self.linkNames[name], self.objName[L.name], UR5e.linkOf[name] = L.name, name, {arm = self, link = L.name}
            LE3EventManager.subscribe("EVT_ON_COLLISION__" .. name, self.name, function(data)
                local other = (data.objectA == name) and data.objectB or data.objectA
                if UR5e.linkOf[other] then return end -- link-to-link contacts come from the mesh test in update()
                local link = self.linkNames[other] -- set when the other body is one of this robot's own links
                if not link then
                    if not UR5e.ignore[other] then self.hits[L.name] = other end
                elseif L.name < link and not self.allowed[L.name .. "|" .. link] then -- each pair once (both links get the event)
                    self.selfHits[L.name .. "|" .. link] = true
                end
            end)
        end
    end
    for _, L in ipairs(LINKS) do -- adjacent links touch by construction: never a self-collision
        local p = L.parent
        while p and not LINKS[p].mesh do p = LINKS[p].parent end
        if L.mesh and p then self:allow(L.name, LINKS[p].name) end
    end
    self:update(0)
end

-- Joint space (radians): shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3; gripper: 0 = open (85 mm), 0.8 = closed
function UR5e:setJoints(q1, q2, q3, q4, q5, q6) self.joints = {q1, q2, q3, q4, q5, q6, self.joints[7]} end
function UR5e:setJoint(i, q) self.joints[i] = q end
function UR5e:getJoints() return table.unpack(self.joints, 1, 6) end
function UR5e:setGripper(q) self.joints[7] = q end
function UR5e:getGripper() return self.joints[7] end
function UR5e:getToolPose() return self.tool.p, self.tool.q end -- tool0 (flange) in world: position {x,y,z}, rotation {w,x,y,z}

-- Contacts with objects outside this robot (other robots' links included) found by the last update: {link = "object name", ...}
function UR5e:getCollisions() return self.collisions end
function UR5e:isInCollision() return next(self.collisions) ~= nil end
-- Self-collisions found by the last update: {["linkA|linkB"] = true, ...}, excluding whitelisted pairs
function UR5e:getSelfCollisions() return self.selfCollisions end
function UR5e:isInSelfCollision() return next(self.selfCollisions) ~= nil end
function UR5e:allowCurrentSelfCollisions() for pair in pairs(self.selfCollisions) do self:allow(pair:match("^(.-)|(.*)$")) end end
function UR5e:allow(x, y) -- whitelist a pair of this robot's links (by link name)
    if x > y then x, y = y, x end
    self.allowed[x .. "|" .. y] = true
    LE3MeshCollision.ignore(self.objName[x], self.objName[y])
end
-- Last configuration (6 joints + gripper) that an update found free of any collision
function UR5e:getLastFree() return {table.unpack(self.lastFree)} end

function UR5e:update(deltaTime)
    self.collisions, self.hits, self.selfCollisions, self.selfHits = self.hits, {}, self.selfHits, {}
    -- contacts of the configuration applied by the previous update: Bullet's from the physics step, the exact ones now
    if self.applied then self:meshContacts() end
    if self.applied and not self:isInCollision() and not self:isInSelfCollision() then self.lastFree = self.applied end
    self.applied = {table.unpack(self.joints)}
    local px, py, pz = LE3Transform.get_position(self.transform)
    local poses, T = UR5e.fk({p = {px, py, pz}, q = {LE3Transform.get_rotation(self.transform)}}, self.joints)
    self.tool = T[TOOL0]
    for i, link in pairs(self.links) do
        local p, q = poses[i].p, poses[i].q
        LE3Transform.set_position(link.transform, p[1], p[2], p[3])
        LE3Transform.set_rotation(link.transform, q[1], q[2], q[3], q[4])
        LE3MeshCollision.set_pose(link.obj, p[1], p[2], p[3], q[1], q[2], q[3], q[4])
        if LE3PhysicsComponent.is_kinematic(link.physics) then -- Bullet only learns the new pose through warp
            LE3PhysicsComponent.warp(link.physics, p[1], p[2], p[3], q[1], q[2], q[3], q[4])
        else
            LE3PhysicsComponent.set_kinematic(link.physics, true) -- no-op until the rigid body exists (first frame)
            LE3PhysicsComponent.set_margin(link.physics, UR5e.margin)
        end
    end
end

-- Exact contacts of this robot's visual meshes (at their current poses) with its own links and other robots' links
function UR5e:meshContacts()
    local pr = LE3MeshCollision.pairs()
    for k = 1, #pr, 2 do
        local na, nb = pr[k], pr[k + 1]
        local a, b = UR5e.linkOf[na], UR5e.linkOf[nb]
        if a and b and a.arm ~= self then a, b, nb = b, a, na end -- make `a` one of this robot's links
        if a and b and a.arm == self then
            if b.arm ~= self then self.collisions[a.link] = nb
            else
                local key = a.link < b.link and a.link .. "|" .. b.link or b.link .. "|" .. a.link
                if not self.allowed[key] then self.selfCollisions[key] = true end
            end
        end
    end
end
