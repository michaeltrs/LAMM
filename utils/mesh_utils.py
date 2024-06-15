import trimesh


def get_region_boundaries(template, fpid):
    mesh = trimesh.load(template)
    faces = mesh.faces

    v2r = {}
    for reg, vals in fpid.items():
        for v in vals:
            if v not in v2r:
                v2r[v] = reg

    bounds_dict = {k: [] for k in fpid.keys()}
    bounds = []
    for face in faces:
        regs = [v2r[f] for f in face]
        if len(set(regs)) > 1:
            bounds.extend(face)
            for f in face:
                bounds_dict[v2r[f]].append(f)

    for k, v in bounds_dict.items():
        bounds_dict[k] = list(set(v))

    return bounds_dict
