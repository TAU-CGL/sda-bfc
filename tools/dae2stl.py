#!/usr/bin/env python3
"""Bake URDF visual/collision Collada meshes into binary STL files that LightEngine3 can load directly.

usage: tools/dae2stl.py <in dir with *.dae> <out dir> [--scale S] [--merge] [--strip PREFIX] [--prefix PREFIX]
  default : one STL per (file, material), named <name>_<Label>.stl (Label = Collada material name, or a color label)
  --merge : one STL per file, all materials merged (for collision meshes)
  --scale : URDF mesh scale (e.g. 0.001 for meshes authored in millimetres)
  --strip / --prefix : rewrite the output base name (e.g. --strip robotiq_arg2f_85_ --prefix rq_)

Node matrices are applied; the Collada up_axis is ignored, exactly like ROS does (visual and collision meshes with
different up_axis tags share one link frame). Normals are flat per face, so STL loses nothing.
Sources: https://github.com/Daniella1/urdf_files_dataset (ur_description/meshes/ur5e/visual, robotiq_2f_85_gripper_visualization/meshes)
"""
import sys, struct, glob, os, argparse
import numpy as np
import xml.etree.ElementTree as ET

NS = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
COLOR_LABELS = {(0.08, 0.08, 0.08): 'RobotiqBlack', (0.7, 0.7, 0.7): 'RobotiqGrey'}  # Robotiq files only name materials "material0/100"

def floats(el): return np.array(el.text.split(), dtype=np.float64)

def label_for(root, material_symbol):
    mats = {m.get('id'): m for m in root.findall('.//c:library_materials/c:material', NS)}
    effects = {e.get('id'): e for e in root.findall('.//c:library_effects/c:effect', NS)}
    binds = {b.get('symbol'): b.get('target')[1:] for b in root.findall('.//c:instance_material', NS)}
    mid = binds.get(material_symbol, material_symbol)
    if mid in mats and mid.endswith('-material'): return mid[:-len('-material')]
    if mid in mats:
        d = effects[mats[mid].find('.//c:instance_effect', NS).get('url')[1:]].find('.//c:diffuse/c:color', NS)
        if d is not None: return COLOR_LABELS.get(tuple(round(v, 2) for v in floats(d)[:3]), 'Grey%d' % round(100 * floats(d)[:3].mean()))
    return 'RobotiqBlack'

def convert(dae_path, out_dir, scale, merge, strip, prefix):
    root = ET.parse(dae_path).getroot()
    name = prefix + os.path.splitext(os.path.basename(dae_path))[0].replace(strip, '', 1)
    sources = {s.get('id'): floats(s.find('c:float_array', NS)).reshape(-1, 3) for s in root.iter('{%s}source' % NS['c'])}
    for v in root.iter('{%s}vertices' % NS['c']):  # <vertices> aliases the POSITION source
        sources[v.get('id')] = sources[v.find("c:input[@semantic='POSITION']", NS).get('source')[1:]]
    geoms = {g.get('id'): g for g in root.findall('.//c:library_geometries/c:geometry', NS)}
    parts = {}  # label -> list of (N,3,3) triangle arrays
    for node in root.findall('.//c:library_visual_scenes//c:node', NS):
        ig = node.find('c:instance_geometry', NS)
        if ig is None: continue
        m = node.find('c:matrix', NS)
        M = floats(m).reshape(4, 4) if m is not None else np.eye(4)
        for prim in geoms[ig.get('url')[1:]].find('c:mesh', NS):
            kind = prim.tag.split('}')[1]
            if kind not in ('polylist', 'triangles'): continue
            inputs = {i.get('semantic'): (int(i.get('offset')), i.get('source')[1:]) for i in prim.findall('c:input', NS)}
            stride = max(o for o, _ in inputs.values()) + 1
            idx = np.array(prim.find('c:p', NS).text.split(), dtype=np.int64).reshape(-1, stride)
            pos = sources[inputs['VERTEX'][1]][idx[:, inputs['VERTEX'][0]]]
            pos = ((M[:3, :3] @ pos.T).T + M[:3, 3]) * scale
            if kind == 'triangles':
                tris = np.arange(len(pos)).reshape(-1, 3)
            else:  # fan-triangulate polygons
                tris, start = [], 0
                for n in np.array(prim.find('c:vcount', NS).text.split(), dtype=np.int64):
                    for k in range(1, n - 1): tris.append((start, start + k, start + k + 1))
                    start += n
                tris = np.array(tris)
            label = '' if merge else label_for(root, prim.get('material'))
            parts.setdefault(label, []).append(pos[tris])
    for label, chunks in parts.items():
        tri = np.vstack(chunks)  # (N,3,3)
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
        rec = np.zeros(len(tri), dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
        rec['n'], rec['v'] = nrm, tri
        out = os.path.join(out_dir, name + ('_' + label if label else '') + '.stl')
        with open(out, 'wb') as f:
            f.write(b'baked by tools/dae2stl.py'.ljust(80, b'\0') + struct.pack('<I', len(tri)) + rec.tobytes())
        print(f'{out}: {len(tri)} triangles, bbox min {tri.reshape(-1, 3).min(0).round(4)} max {tri.reshape(-1, 3).max(0).round(4)}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('in_dir'); ap.add_argument('out_dir'); ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--merge', action='store_true'); ap.add_argument('--strip', default=''); ap.add_argument('--prefix', default='')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for dae in sorted(glob.glob(os.path.join(a.in_dir, '*.dae'))): convert(dae, a.out_dir, a.scale, a.merge, a.strip, a.prefix)
