"""PLY con colores de visualización, etiquetas y RGB original, sin reordenar puntos."""

import numpy as np


def write_labeled_ply(path, xyz, normals, original_rgb, display_rgb, labels, classes):
    fields = [
        (property_name, "<f4") for property_name in ["x", "y", "z", "nx", "ny", "nz"]
    ]
    fields += [
        (property_name, "u1")
        for property_name in [
            "red",
            "green",
            "blue",
            "original_red",
            "original_green",
            "original_blue",
        ]
    ]
    fields += [("instance_id", "<i4"), ("semantic_id", "u1")]
    records = np.empty(len(xyz), dtype=np.dtype(fields))
    for axis_index, property_name in enumerate(["x", "y", "z"]):
        records[property_name] = xyz[:, axis_index]
    for axis_index, property_name in enumerate(["nx", "ny", "nz"]):
        records[property_name] = normals[:, axis_index]
    for axis_index, property_name in enumerate(["red", "green", "blue"]):
        records[property_name] = np.rint(
            np.clip(display_rgb[:, axis_index], 0, 1) * 255
        ).astype("u1")
        records["original_" + property_name] = np.rint(
            np.clip(original_rgb[:, axis_index], 0, 1) * 255
        ).astype("u1")
    records["instance_id"] = labels
    records["semantic_id"] = classes
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Original point order preserved; unknown instance=-1 semantic=0",
        f"element vertex {len(xyz)}",
    ]
    mapping = {"f": "float", "i": "int", "u": "uchar"}
    for name, property_dtype in fields:
        header.append(f"property {mapping[np.dtype(property_dtype).kind]} {name}")
    header.append("end_header")
    with open(path, "wb") as stream:
        stream.write(("\n".join(header) + "\n").encode())
        records.tofile(stream)


def read_records(path):
    types = {"float": "<f4", "int": "<i4", "uchar": "u1"}
    properties = []
    with open(path, "rb") as stream:
        while True:
            line = stream.readline().decode().strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            if line.startswith("property"):
                (_, property_type, name) = line.split()
                properties.append((name, types[property_type]))
            if line == "end_header":
                break
        return np.fromfile(stream, dtype=np.dtype(properties), count=count)
