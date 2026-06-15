import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import racetrack

# high field low current
straight_types = ["Brown", "OM", "OM", "OM", "Brown"]
center_types = [
    "Blue",
    "12pan",
    "BlueCenter",
    "12panCenter",
    "BlueCenter",
    "12pan",
    "Blue",
]

Mirror_Length = 1.5
Stellerator_Radius = 0.43
# filename = "medium_Lm_1p5.csv"
filename = "../test_files/racetrack_12pan.csv"

rt = racetrack.racetrack(
    Mirror_Length,
    Stellerator_Radius / 2,
    straight_types,
    center_types,
    filename=filename,
)
rt.build_coils()
rt.write_csv()
