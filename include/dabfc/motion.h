#ifndef DABFC_MOTION_H_
#define DABFC_MOTION_H_

namespace dabfc {

// A motion of the arms of a DualArmSimulator, advanced by its caller, who sets the pace: offline,
// step(infinity) until it returns false; online, step(speed * dt) once per frame
class Motion {
public:
    Motion() = default;
    Motion(const Motion&) = delete;
    Motion& operator=(const Motion&) = delete;
    Motion(Motion&&) = delete;
    Motion& operator=(Motion&&) = delete;
    virtual ~Motion() = default;

    // Moves by at most distance (the largest joint change, rad); false once the motion is over
    virtual bool step(double distance) = 0;
};

} // namespace dabfc

#endif // DABFC_MOTION_H_
