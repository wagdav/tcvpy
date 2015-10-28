from tcv.geom import StaticParameters
from tcv.geom import VesselDrawer

from numpy.testing import assert_equal


# fixture static parameters
vessel_coordinates = {
    'R_in': [0, 1, 2, 3],
    'Z_in': [0, 1, 2, 3],
    'R_out': [0, 10, 20, 30],
    'Z_out': [0, 10, 20, 30]
}
tcv = StaticParameters({'vessel': vessel_coordinates})


def test_get_coordinates():
    assert_equal(
        tcv.get_coordinates('vessel', 'R_in'),
        vessel_coordinates['R_in'])

    desired = (
        vessel_coordinates['R_in'], vessel_coordinates['R_in'])
    assert_equal(tcv.get_coordinates('vessel', 'R_in, Z_in'), desired)

    assert_equal(tcv.get_coordinates('vessel'), vessel_coordinates)


def test_vessel_drawer():
    vessel = VesselDrawer({'vessel': vessel_coordinates})

    assert_equal(vessel.has_tiles_coordinates(), False)

    vertices, codes = vessel._inside()
    assert_equal(vertices[:-1],
                 [(x, y) for x, y in zip(vessel_coordinates['R_in'],
                                         vessel_coordinates['Z_in'])])
    assert_equal(vertices[-1], vertices[0])
    assert_equal(codes[0], 1)  # Path.MOVETO
    assert_equal(codes[1:], (len(codes) - 1) * [2])  # Path.MOVETO

    vertices, codes = vessel._outside()
    assert_equal(vertices[:-1][::-1],
                 [(x, y) for x, y in zip(vessel_coordinates['R_out'],
                                         vessel_coordinates['Z_out'])])
    assert_equal(vertices[-1], vertices[0])
    assert_equal(codes[0], 1)  # Path.MOVETO
    assert_equal(codes[1:], (len(codes) - 1) * [2])  # Path.MOVETO
