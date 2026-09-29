-- ImGui panel with joint and gripper sliders for two UR5e arms. Add as a ScriptObject with Classname = "ArmsController".
-- Dragging moves an arm live. On release, if that arm is in a self-collision or touches anything else (the other arm,
-- the world), it bisects between the last collision-free configuration and the released one and settles on the
-- configuration where the links just touch (to about 0.1 deg).
ArmsController = LE3ScriptObject:new()
ArmsController.arms = {"r1", "r2"} -- names of the UR5e script objects in the scene

local DEG = 180 / math.pi
local SLIDERS = { -- label, min, max (degrees for joints, radians for the gripper)
    {"shoulder_pan", -360, 360}, {"shoulder_lift", -360, 360}, {"elbow", -180, 180},
    {"wrist_1", -360, 360}, {"wrist_2", -360, 360}, {"wrist_3", -360, 360}, {"gripper", 0, 0.8},
}
local CALIBRATION_FRAMES = 120 -- startup: close and reopen the grippers while whitelisting every self-contact (adjacent links, finger linkage)
local VERDICT_FRAMES = 2 -- a configuration applied in one frame is judged by the physics two frames later (see UR5e:update)
local BISECT_STEPS = 10 -- release in collision: halve the bracket this many times (a 90 deg drag ends within 0.1 deg of contact)

local function apply(arm, q) arm:setJoints(table.unpack(q, 1, 6)); arm:setGripper(q[7]) end
local function midpoint(a, b) local m = {} for i = 1, #a do m[i] = 0.5 * (a[i] + b[i]) end return m end

function ArmsController:init() self.frame, self.state = 0, {} end

function ArmsController:blocked(arm) return arm:isInCollision() or arm:isInSelfCollision() end

-- Runs after a slider release until the arm rests on a collision-free configuration. s.wait counts frames until the
-- physics verdict on the last applied configuration; s.lo/s.hi bracket the contact boundary (free/colliding).
function ArmsController:settle(arm, s, blocked)
    if s.wait > 0 then s.wait = s.wait - 1 return end
    if s.hi then if blocked then s.hi = s.q else s.lo = s.q end          -- verdict on the last midpoint
    elseif blocked then s.lo, s.hi, s.step = arm:getLastFree(), s.q, 0  -- released in collision: start bisecting
    else s.wait = nil return end                                         -- released free: nothing to do
    s.step = s.step + 1
    if s.step > BISECT_STEPS then s.q, s.hi, s.wait = s.lo, nil, nil else s.q, s.wait = midpoint(s.lo, s.hi), VERDICT_FRAMES end
    apply(arm, s.q)
end

function ArmsController:update(deltaTime)
    self.frame = self.frame + 1
    ImGui.SetNextWindowSize(440, 500)
    ImGui.Begin("Arms")
    for _, name in ipairs(self.arms) do
        local arm = UR5e._refs[name]
        local s = self.state[name]
        if not arm then
            ImGui.Text(name .. ": no UR5e object with this name")
        elseif self.frame <= CALIBRATION_FRAMES then
            self.state[name] = s or {q = arm:getLastFree()}
            arm:setGripper(0.4 * (1 - math.cos(2 * math.pi * self.frame / CALIBRATION_FRAMES)))
            arm:allowCurrentSelfCollisions()
            ImGui.Text(name .. ": learning self-collision pairs...")
        else
            local blocked = self:blocked(arm)
            if s.wait then self:settle(arm, s, blocked) end
            ImGui.TextColored(blocked and 1 or 0.4, blocked and 0.4 or 1, 0.4, 1, name .. (blocked and "   COLLISION" or "   ok"))
            for i, sl in ipairs(SLIDERS) do
                local scale = i < 7 and DEG or 1
                local v, changed = ImGui.SliderFloat(name .. " " .. sl[1], s.q[i] * scale, sl[2], sl[3], i < 7 and "%.1f deg" or "%.2f rad")
                if changed then s.q[i] = v / scale; s.hi, s.wait = nil, nil; apply(arm, s.q) end -- grabbing a slider also cancels a settle in progress
                if ImGui.IsItemDeactivatedAfterEdit() then s.wait = VERDICT_FRAMES end
            end
        end
        ImGui.Separator()
    end
    ImGui.End()
end
