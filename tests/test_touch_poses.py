import numpy as np
import pytest

from sda_bfc import CylinderPose, UR5e

TOUCH_POSES = [
    ([np.float64(-1.2169), np.float64(0.7579), np.float64(1.1118), np.float64(-1.9127), np.float64(-1.1849), np.float64(0.4488)], [np.float64(-1.1239), np.float64(2.1736), np.float64(1.4927), np.float64(-1.5344), np.float64(-0.7174), np.float64(0.6371)]),
    ([np.float64(-0.9916), np.float64(-0.7073), np.float64(0.6476), np.float64(1.7401), np.float64(-0.0626), np.float64(0.5265)], [np.float64(-1.5439), np.float64(-2.1965), np.float64(0.681), np.float64(2.6306), np.float64(-2.5348), np.float64(-2.5649)]),
    ([np.float64(2.7261), np.float64(2.0194), np.float64(2.7965), np.float64(-1.6985), np.float64(-0.6912), np.float64(2.8698)], [np.float64(1.5799), np.float64(0.3162), np.float64(-0.8449), np.float64(0.0105), np.float64(3.0781), np.float64(1.9694)]),
    ([np.float64(1.3351), np.float64(2.7837), np.float64(-0.2745), np.float64(-2.8763), np.float64(1.9566), np.float64(2.5491)], [np.float64(-0.4784), np.float64(2.7684), np.float64(2.3405), np.float64(0.5237), np.float64(-1.2146), np.float64(0.5891)]),
    ([np.float64(2.6055), np.float64(-3.0891), np.float64(0.8444), np.float64(0.0712), np.float64(2.9923), np.float64(-2.6337)], [np.float64(-2.0943), np.float64(-2.5561), np.float64(0.0261), np.float64(0.6745), np.float64(-0.7761), np.float64(1.8969)]),
    ([np.float64(2.3494), np.float64(1.5549), np.float64(2.6593), np.float64(1.6145), np.float64(-0.6115), np.float64(-3.042)], [np.float64(-1.3889), np.float64(2.9262), np.float64(0.4087), np.float64(0.3865), np.float64(2.7902), np.float64(-2.9034)]),
    ([np.float64(2.2227), np.float64(-2.6863), np.float64(-2.7743), np.float64(-1.0776), np.float64(0.2952), np.float64(-0.5785)], [np.float64(1.574), np.float64(-0.1289), np.float64(-1.0333), np.float64(-2.21), np.float64(1.485), np.float64(-2.178)]),
    ([np.float64(-0.6809), np.float64(0.2451), np.float64(-0.1461), np.float64(-0.8713), np.float64(0.4254), np.float64(3.1362)], [np.float64(1.2466), np.float64(-0.5402), np.float64(1.9957), np.float64(-1.9415), np.float64(-0.6972), np.float64(-1.5066)]),
    ([np.float64(1.9024), np.float64(-2.7764), np.float64(-0.566), np.float64(1.1915), np.float64(0.5848), np.float64(-2.3918)], [np.float64(-2.0671), np.float64(-1.4982), np.float64(-1.9548), np.float64(2.873), np.float64(-1.3195), np.float64(-1.1516)]),
    ([np.float64(-2.3728), np.float64(0.0892), np.float64(1.4439), np.float64(-2.9251), np.float64(0.045), np.float64(2.5481)], [np.float64(2.3624), np.float64(-0.1589), np.float64(0.5187), np.float64(-2.3854), np.float64(1.2113), np.float64(0.2955)]),
    ([np.float64(1.3195), np.float64(-2.9155), np.float64(-0.2501), np.float64(-0.9823), np.float64(1.4037), np.float64(3.0532)], [np.float64(2.6554), np.float64(-0.7401), np.float64(-0.3288), np.float64(-2.5141), np.float64(-2.4375), np.float64(2.8185)]),
    ([np.float64(1.9217), np.float64(-2.6457), np.float64(-0.9143), np.float64(-1.2908), np.float64(2.0616), np.float64(-1.4153)], [np.float64(-1.762), np.float64(-1.7651), np.float64(-1.9486), np.float64(-2.9455), np.float64(-1.588), np.float64(-3.0084)]),
    ([np.float64(1.6415), np.float64(2.9892), np.float64(0.7412), np.float64(-1.4949), np.float64(-1.2196), np.float64(-0.315)], [np.float64(3.1006), np.float64(-0.6151), np.float64(-2.1793), np.float64(2.6961), np.float64(-1.9024), np.float64(2.57)]),
    ([np.float64(-1.472), np.float64(0.7358), np.float64(-1.12), np.float64(-2.7418), np.float64(-1.6359), np.float64(0.0314)], [np.float64(-0.47), np.float64(-2.4559), np.float64(-2.4822), np.float64(-1.1473), np.float64(-1.4883), np.float64(1.6819)]),
    ([np.float64(2.51), np.float64(1.8731), np.float64(0.8524), np.float64(2.7258), np.float64(0.0075), np.float64(-2.5386)], [np.float64(-1.4821), np.float64(2.8976), np.float64(-0.8983), np.float64(0.2446), np.float64(0.7594), np.float64(-0.7674)]),
    ([np.float64(-0.5829), np.float64(0.7495), np.float64(-1.0101), np.float64(0.4185), np.float64(-3.0358), np.float64(1.0068)], [np.float64(-1.6005), np.float64(-2.6769), np.float64(-1.6395), np.float64(2.3809), np.float64(0.8826), np.float64(0.1234)]),
    ([np.float64(-0.612), np.float64(0.4468), np.float64(-2.7229), np.float64(-0.1148), np.float64(0.726), np.float64(0.3832)], [np.float64(1.5674), np.float64(-0.6547), np.float64(1.52), np.float64(0.8129), np.float64(1.5435), np.float64(2.6355)]),
    ([np.float64(-0.8098), np.float64(-0.1181), np.float64(1.4041), np.float64(-2.745), np.float64(-2.9693), np.float64(1.203)], [np.float64(1.4715), np.float64(-0.6243), np.float64(2.1278), np.float64(-2.6723), np.float64(-1.4648), np.float64(1.1391)]),
    ([np.float64(0.8648), np.float64(2.3167), np.float64(-0.8076), np.float64(0.8603), np.float64(0.2889), np.float64(-2.0273)], [np.float64(-0.9923), np.float64(2.8203), np.float64(-0.1576), np.float64(-2.6881), np.float64(0.7252), np.float64(2.7423)]),
    ([np.float64(2.3939), np.float64(-2.6383), np.float64(-0.3645), np.float64(-2.8896), np.float64(2.041), np.float64(2.8609)], [np.float64(1.2582), np.float64(-0.668), np.float64(-0.1891), np.float64(-2.5511), np.float64(2.6818), np.float64(0.4304)]),
]

BASE_OFFSET = np.array([-0.24, 0.73, -0.25, 0, 0, 0])

DH_LINK_INDEX = 3
RADII_LINK_INDEX = 4
TOUCH_TOLERANCE = 2e-3


def rotation_vector_to_matrix(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def base_transform():
    T = np.eye(4)
    T[:3, :3] = rotation_vector_to_matrix(BASE_OFFSET[3:])
    T[:3, 3] = BASE_OFFSET[:3]
    return T


@pytest.mark.parametrize("q1,q2", TOUCH_POSES)
def test_touching_forearms_have_zero_signed_distance(q1, q2):
    robot = UR5e()
    radius = robot.get_link_radius(RADII_LINK_INDEX)
    T1 = robot.get_cylinder_transform(DH_LINK_INDEX, np.array(q1))
    T2 = base_transform() @ robot.get_cylinder_transform(DH_LINK_INDEX, np.array(q2))
    c1 = CylinderPose.from_se3(T1, radius)
    c2 = CylinderPose.from_se3(T2, radius)
    assert c1.signed_distance(c2) == pytest.approx(0.0, abs=TOUCH_TOLERANCE)
