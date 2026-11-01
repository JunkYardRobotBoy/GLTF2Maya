"""GLTF2Maya - a glTF 2.0 / GLB importer plug-in for Autodesk Maya.

Companion to the Maya2glTF exporter (https://github.com/JunkYardRobotBoy/Maya2glTF_Update).
Pure Python, no third-party dependencies. Tested target: Maya 2026 / 2027 (Python 3).

Imports:
  * node hierarchy (transforms / joints), TRS or matrix
  * meshes with normals, up to two UV sets, vertex colours, multi-material primitives
  * shared meshes and EXT_mesh_gpu_instancing as Maya instances
  * materials as standardSurface networks (base colour, metal/rough, normal, emissive,
    alpha, KHR_texture_transform, KHR_materials_emissive_strength / transmission / ior / clearcoat)
  * skins as skinClusters with the glTF inverse bind matrices
  * morph targets as blendShapes (names from mesh.extras.targetNames)
  * animations (TRS + morph weights, STEP / LINEAR / CUBICSPLINE), all clips concatenated
  * cameras and KHR_lights_punctual lights

Usage (inside Maya):
    cmds.loadPlugin('GLTF2Maya.py')
    cmds.gltf2maya(file=r'C:/path/model.glb', scale=100)
or
    import GLTF2Maya; GLTF2Maya.show_ui()
or File > Import with the "glTF" file type.
"""

import os
import re
import sys
import json
import math
import base64
import struct

try:
    import numpy as _np
except Exception:  # numpy is bundled with Maya 2025+, but stay portable
    _np = None

__version__ = '0.1.0'

# ---------------------------------------------------------------------------
# glTF parsing (no Maya dependency, so it can be unit-tested outside Maya)
# ---------------------------------------------------------------------------

GLB_MAGIC = 0x46546C67  # 'glTF'
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMPONENT_FMT = {
    5120: ('b', 1, 127.0),
    5121: ('B', 1, 255.0),
    5122: ('h', 2, 32767.0),
    5123: ('H', 2, 65535.0),
    5125: ('I', 4, 4294967295.0),
    5126: ('f', 4, None),
}
TYPE_COUNT = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}
MIME_EXT = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp', 'image/ktx2': '.ktx2',
            'image/bmp': '.bmp'}


class GltfError(Exception):
    pass


def _group(flat, n):
    if n == 1:
        return list(flat)
    it = iter(flat)
    return list(zip(*([it] * n)))


class GltfDocument(object):
    """Loads a .gltf or .glb file and gives typed access to accessors and images."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.base_dir = os.path.dirname(self.path)
        self.name = os.path.splitext(os.path.basename(self.path))[0]
        self.gltf = None
        self._glb_bin = None
        self._buffers = {}
        self._accessor_cache = {}
        self.warnings = []
        self._load()

    # -- loading -----------------------------------------------------------
    def _load(self):
        with open(self.path, 'rb') as fh:
            data = fh.read()
        if len(data) >= 12 and struct.unpack_from('<I', data, 0)[0] == GLB_MAGIC:
            self._parse_glb(data)
        else:
            self.gltf = json.loads(data.decode('utf-8-sig'))
        asset = self.gltf.get('asset', {})
        major = str(asset.get('version', '2.0')).split('.')[0]
        if major != '2':
            raise GltfError('Unsupported glTF version %s (only 2.x is supported)' % asset.get('version'))
        for ext in self.gltf.get('extensionsRequired', []):
            if ext in ('KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'KHR_texture_basisu'):
                self.warn('Required extension %s is not supported; affected data will be skipped' % ext)

    def _parse_glb(self, data):
        magic, version, length = struct.unpack_from('<III', data, 0)
        if version != 2:
            raise GltfError('Unsupported GLB container version %d' % version)
        offset = 12
        json_chunk = None
        while offset + 8 <= min(length, len(data)):
            chunk_len, chunk_type = struct.unpack_from('<II', data, offset)
            offset += 8
            chunk = data[offset:offset + chunk_len]
            offset += chunk_len
            if chunk_type == CHUNK_JSON and json_chunk is None:
                json_chunk = chunk
            elif chunk_type == CHUNK_BIN and self._glb_bin is None:
                self._glb_bin = chunk
        if json_chunk is None:
            raise GltfError('GLB file has no JSON chunk')
        self.gltf = json.loads(json_chunk.decode('utf-8'))

    def warn(self, msg):
        self.warnings.append(msg)

    # -- buffers -------------------------------------------------------------
    def _resolve_uri(self, uri):
        """Returns bytes for a data: URI or a relative/absolute file URI."""
        if uri.startswith('data:'):
            header, _, payload = uri.partition(',')
            if ';base64' in header:
                return base64.b64decode(payload)
            try:
                from urllib.parse import unquote_to_bytes
            except ImportError:  # pragma: no cover
                from urllib import unquote as unquote_to_bytes
            return unquote_to_bytes(payload)
        try:
            from urllib.parse import unquote
        except ImportError:  # pragma: no cover
            from urllib import unquote
        rel = unquote(uri)
        path = rel if os.path.isabs(rel) else os.path.join(self.base_dir, rel)
        with open(path, 'rb') as fh:
            return fh.read()

    def buffer(self, index):
        if index in self._buffers:
            return self._buffers[index]
        buf = self.gltf['buffers'][index]
        if 'uri' in buf:
            data = self._resolve_uri(buf['uri'])
        else:
            if self._glb_bin is None:
                raise GltfError('Buffer %d has no uri and the file has no BIN chunk' % index)
            data = self._glb_bin
        self._buffers[index] = data
        return data

    def buffer_view(self, index):
        bv = self.gltf['bufferViews'][index]
        data = self.buffer(bv['buffer'])
        start = bv.get('byteOffset', 0)
        return data[start:start + bv['byteLength']]

    # -- accessors -----------------------------------------------------------
    def _unpack(self, data, start, count, ncomp, comp_type, stride=None):
        fmt, csize, _ = COMPONENT_FMT[comp_type]
        elem_size = ncomp * csize
        if not stride or stride == elem_size:
            total = count * ncomp
            if _np is not None:
                arr = _np.frombuffer(data, dtype='<' + fmt, count=total, offset=start)
                flat = arr.tolist()
            else:
                flat = struct.unpack_from('<%d%s' % (total, fmt), data, start)
            return _group(flat, ncomp)
        out = []
        efmt = '<%d%s' % (ncomp, fmt)
        for i in range(count):
            v = struct.unpack_from(efmt, data, start + i * stride)
            out.append(v if ncomp > 1 else v[0])
        return out

    def accessor(self, index):
        """Returns a list of scalars (SCALAR) or tuples (VECn / MATn)."""
        if index in self._accessor_cache:
            return self._accessor_cache[index]
        acc = self.gltf['accessors'][index]
        count = acc['count']
        ncomp = TYPE_COUNT[acc['type']]
        comp_type = acc['componentType']
        if 'bufferView' in acc:
            bv = self.gltf['bufferViews'][acc['bufferView']]
            data = self.buffer(bv['buffer'])
            start = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
            values = self._unpack(data, start, count, ncomp, comp_type, bv.get('byteStride'))
        else:
            zero = (0,) * ncomp if ncomp > 1 else 0
            values = [zero] * count
        sparse = acc.get('sparse')
        if sparse:
            n = sparse['count']
            ibv = self.gltf['bufferViews'][sparse['indices']['bufferView']]
            idata = self.buffer(ibv['buffer'])
            istart = ibv.get('byteOffset', 0) + sparse['indices'].get('byteOffset', 0)
            idx = self._unpack(idata, istart, n, 1, sparse['indices']['componentType'])
            vbv = self.gltf['bufferViews'][sparse['values']['bufferView']]
            vdata = self.buffer(vbv['buffer'])
            vstart = vbv.get('byteOffset', 0) + sparse['values'].get('byteOffset', 0)
            vals = self._unpack(vdata, vstart, n, ncomp, comp_type)
            values = list(values)
            for i, v in zip(idx, vals):
                values[i] = v
        if acc.get('normalized') and COMPONENT_FMT[comp_type][2]:
            m = COMPONENT_FMT[comp_type][2]
            signed = comp_type in (5120, 5122)
            if ncomp == 1:
                values = [max(v / m, -1.0) if signed else v / m for v in values]
            else:
                values = [tuple(max(c / m, -1.0) if signed else c / m for c in v) for v in values]
        self._accessor_cache[index] = values
        return values

    # -- images --------------------------------------------------------------
    def image_path(self, index, out_dir):
        """Returns a path on disk for image `index`, extracting embedded images into out_dir."""
        img = self.gltf['images'][index]
        uri = img.get('uri')
        if uri and not uri.startswith('data:'):
            try:
                from urllib.parse import unquote
            except ImportError:  # pragma: no cover
                from urllib import unquote
            rel = unquote(uri)
            return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(self.base_dir, rel))
        if uri:
            header = uri.split(',', 1)[0]
            mime = header[5:].split(';')[0]
            data = self._resolve_uri(uri)
        elif 'bufferView' in img:
            mime = img.get('mimeType', 'image/png')
            data = self.buffer_view(img['bufferView'])
        else:
            raise GltfError('Image %d has neither uri nor bufferView' % index)
        ext = MIME_EXT.get(mime)
        if ext is None:
            ext = '.png' if data[:4] == b'\x89PNG' else '.jpg'
        name = sanitize_name(img.get('name') or 'image_%d' % index)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        path = os.path.join(out_dir, name + ext)
        if not os.path.exists(path) or os.path.getsize(path) != len(data):
            with open(path, 'wb') as fh:
                fh.write(data)
        return path

    # -- convenience ---------------------------------------------------------
    def nodes(self):
        return self.gltf.get('nodes', [])

    def scene_roots(self):
        scenes = self.gltf.get('scenes')
        if scenes:
            scene = scenes[self.gltf.get('scene', 0)]
            return list(scene.get('nodes', []))
        children = set()
        for n in self.nodes():
            children.update(n.get('children', []))
        return [i for i in range(len(self.nodes())) if i not in children]

    def skin_joint_set(self):
        joints = set()
        for skin in self.gltf.get('skins', []):
            joints.update(skin['joints'])
        return joints


def sanitize_name(name, fallback='node'):
    name = re.sub(r'[^A-Za-z0-9_]', '_', str(name or '')).strip('_')
    if not name:
        name = fallback
    if name[0].isdigit():
        name = '_' + name
    return name


def triangulate(indices, mode):
    """Converts TRIANGLES / TRIANGLE_STRIP / TRIANGLE_FAN index lists to a flat triangle list."""
    if mode == 4:
        n = len(indices) - len(indices) % 3
        return list(indices[:n])
    tris = []
    if mode == 5:
        for i in range(len(indices) - 2):
            a, b, c = indices[i], indices[i + 1], indices[i + 2]
            if i % 2:
                tris.extend((b, a, c))
            else:
                tris.extend((a, b, c))
    elif mode == 6:
        for i in range(1, len(indices) - 1):
            tris.extend((indices[0], indices[i], indices[i + 1]))
    return tris


def quat_to_matrix(q):
    """Row-vector (Maya) 4x4 flat matrix from a glTF quaternion (x, y, z, w)."""
    x, y, z, w = q
    return [
        1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0,
        2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
        2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0,
        0, 0, 0, 1,
    ]


class MeshData(object):
    """Flattened, Maya-ready geometry for one glTF mesh (all primitives merged)."""

    def __init__(self):
        self.positions = []     # [(x,y,z)]
        self.normals = []       # [(x,y,z)] or None when any primitive lacks them
        self.uv0 = []           # [(u,v)] glTF convention (v down)
        self.uv1 = None
        self.colors = None      # [(r,g,b,a)]
        self.joints = None      # per vertex: dict {joint_in_skin: weight}
        self.triangles = []     # flat vertex indices
        self.prim_faces = []    # (material index or None, first_face, face_count)
        self.targets = []       # list of [(dx,dy,dz)] per morph target
        self.target_names = []
        self.has_normals = True
        self.has_uv1 = False
        self.has_colors = False


def build_mesh_data(doc, mesh_index, want_skin=True, want_morphs=True):
    mesh = doc.gltf['meshes'][mesh_index]
    md = MeshData()
    prims = mesh.get('primitives', [])
    n_targets = max([len(p.get('targets', [])) for p in prims] + [0]) if want_morphs else 0
    md.targets = [[] for _ in range(n_targets)]
    md.has_uv1 = any('TEXCOORD_1' in p.get('attributes', {}) for p in prims)
    md.has_colors = any('COLOR_0' in p.get('attributes', {}) for p in prims)
    has_skin = want_skin and any('JOINTS_0' in p.get('attributes', {}) for p in prims)
    if md.has_uv1:
        md.uv1 = []
    if md.has_colors:
        md.colors = []
    if has_skin:
        md.joints = []
    for pi, prim in enumerate(prims):
        attrs = prim.get('attributes', {})
        mode = prim.get('mode', 4)
        if 'POSITION' not in attrs:
            doc.warn('Mesh %d primitive %d has no POSITION, skipped' % (mesh_index, pi))
            continue
        exts = prim.get('extensions', {})
        if 'KHR_draco_mesh_compression' in exts and 'bufferView' not in doc.gltf['accessors'][attrs['POSITION']]:
            doc.warn('Mesh %d primitive %d is Draco compressed, skipped (re-export uncompressed)' % (mesh_index, pi))
            continue
        if mode not in (4, 5, 6):
            doc.warn('Mesh %d primitive %d uses mode %d (points/lines), skipped' % (mesh_index, pi, mode))
            continue
        pos = doc.accessor(attrs['POSITION'])
        nvert = len(pos)
        if 'indices' in prim:
            idx = doc.accessor(prim['indices'])
        else:
            idx = list(range(nvert))
        tris = triangulate(idx, mode)
        base = len(md.positions)
        # drop degenerate triangles, MFnMesh.create rejects them
        clean = []
        for i in range(0, len(tris), 3):
            a, b, c = tris[i], tris[i + 1], tris[i + 2]
            if a != b and b != c and a != c and max(a, b, c) < nvert:
                clean.extend((a + base, b + base, c + base))
        nfaces = len(clean) // 3
        if nfaces == 0:
            continue
        md.positions.extend(tuple(p) for p in pos)
        md.triangles.extend(clean)
        md.prim_faces.append((prim.get('material'), sum(f[2] for f in md.prim_faces), nfaces))
        if 'NORMAL' in attrs and md.has_normals:
            md.normals.extend(tuple(n) for n in doc.accessor(attrs['NORMAL']))
        else:
            md.has_normals = False
            md.normals = None
        if 'TEXCOORD_0' in attrs:
            md.uv0.extend(tuple(t) for t in doc.accessor(attrs['TEXCOORD_0']))
        else:
            md.uv0.extend([(0.0, 0.0)] * nvert)
        if md.has_uv1:
            if 'TEXCOORD_1' in attrs:
                md.uv1.extend(tuple(t) for t in doc.accessor(attrs['TEXCOORD_1']))
            else:
                md.uv1.extend([(0.0, 0.0)] * nvert)
        if md.has_colors:
            if 'COLOR_0' in attrs:
                for c in doc.accessor(attrs['COLOR_0']):
                    md.colors.append((c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0))
            else:
                md.colors.extend([(1.0, 1.0, 1.0, 1.0)] * nvert)
        if has_skin:
            per_vertex = [dict() for _ in range(nvert)]
            k = 0
            while 'JOINTS_%d' % k in attrs and 'WEIGHTS_%d' % k in attrs:
                js = doc.accessor(attrs['JOINTS_%d' % k])
                ws = doc.accessor(attrs['WEIGHTS_%d' % k])
                for v in range(nvert):
                    for j, w in zip(js[v], ws[v]):
                        if w > 0.0:
                            per_vertex[v][int(j)] = per_vertex[v].get(int(j), 0.0) + float(w)
                k += 1
            md.joints.extend(per_vertex)
        for t in range(n_targets):
            targets = prim.get('targets', [])
            if t < len(targets) and 'POSITION' in targets[t]:
                md.targets[t].extend(tuple(d) for d in doc.accessor(targets[t]['POSITION']))
            else:
                md.targets[t].extend([(0.0, 0.0, 0.0)] * nvert)
    names = mesh.get('extras', {}).get('targetNames') or []
    md.target_names = [sanitize_name(names[i] if i < len(names) else 'target%d' % i, 'target%d' % i)
                       for i in range(n_targets)]
    return md


# ---------------------------------------------------------------------------
# Maya side
# ---------------------------------------------------------------------------

try:
    import maya.api.OpenMaya as om
    import maya.api.OpenMayaAnim as oma
    import maya.cmds as cmds
    IN_MAYA = True
except ImportError:
    IN_MAYA = False


def maya_useNewAPI():
    """Tells Maya this plug-in uses the Python API 2.0."""
    pass


class ImportOptions(object):
    def __init__(self):
        self.scale = 1.0
        self.materials = True
        self.skins = True
        self.morphs = True
        self.animation = True
        self.cameras = True
        self.lights = True
        self.normals = True
        self.merge_vertices = False
        self.clip = -1          # -1 = all animations, else index of one animation
        self.textures_dir = None

    @classmethod
    def from_optionvars(cls):
        o = cls()
        if not IN_MAYA:
            return o
        for key, attr, kind in OPTIONVAR_MAP:
            if cmds.optionVar(exists=key):
                val = cmds.optionVar(q=key)
                setattr(o, attr, bool(val) if kind == 'bool' else (float(val) if kind == 'float' else int(val)))
        return o

    def to_optionvars(self):
        for key, attr, kind in OPTIONVAR_MAP:
            val = getattr(self, attr)
            if kind == 'float':
                cmds.optionVar(fv=(key, float(val)))
            else:
                cmds.optionVar(iv=(key, int(val)))


OPTIONVAR_MAP = [
    ('GLTF2Maya_scale', 'scale', 'float'),
    ('GLTF2Maya_materials', 'materials', 'bool'),
    ('GLTF2Maya_skins', 'skins', 'bool'),
    ('GLTF2Maya_morphs', 'morphs', 'bool'),
    ('GLTF2Maya_animation', 'animation', 'bool'),
    ('GLTF2Maya_cameras', 'cameras', 'bool'),
    ('GLTF2Maya_lights', 'lights', 'bool'),
    ('GLTF2Maya_normals', 'normals', 'bool'),
    ('GLTF2Maya_mergeVertices', 'merge_vertices', 'bool'),
    ('GLTF2Maya_clip', 'clip', 'int'),
]

PLACE2D_CONNECTIONS = [
    'coverage', 'translateFrame', 'rotateFrame', 'mirrorU', 'mirrorV', 'stagger', 'wrapU', 'wrapV',
    'repeatUV', 'offset', 'rotateUV', 'noiseUV', 'vertexUvOne', 'vertexUvTwo', 'vertexUvThree',
    'vertexCameraOne',
]


def _mobject(name):
    sel = om.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


def _dagpath(name):
    sel = om.MSelectionList()
    sel.add(name)
    return sel.getDagPath(0)


def _plug(name, attr):
    return om.MFnDependencyNode(_mobject(name)).findPlug(attr, False)


class GltfImporter(object):
    def __init__(self, doc, options):
        self.doc = doc
        self.gltf = doc.gltf
        self.opt = options
        self.node_names = {}        # glTF node index -> Maya transform name
        self.joint_set = doc.skin_joint_set() if options.skins else set()
        self.materials = {}         # glTF material index -> (shadingGroup, [(fileNode, texCoord)])
        self.mesh_cache = {}        # glTF mesh index -> (shape, mesh_data) for instancing
        self.mesh_nodes = {}        # glTF node index -> (transform, shape, blendShape or None)
        self.mesh_nodes_pending = set()
        self.root = None
        self.fps = om.MTime(1.0, om.MTime.kSeconds).asUnits(om.MTime.uiUnit())
        self.textures_dir = options.textures_dir or os.path.join(doc.base_dir, doc.name + '_textures')
        self.clip_ranges = []

    # -- entry -----------------------------------------------------------------
    def run(self):
        doc = self.doc
        self.root = cmds.ls(cmds.createNode('transform', name=sanitize_name(doc.name, 'glTF')), long=True)[0]
        s = float(self.opt.scale)
        if s != 1.0:
            cmds.setAttr(self.root + '.scale', s, s, s)
        if om.MGlobal.isZAxisUp():
            cmds.setAttr(self.root + '.rotateX', 90)
        for ni in doc.scene_roots():
            self._build_node(ni, self.root)
        for ni in sorted(self.mesh_nodes_pending):
            self._build_mesh_for_node(ni)
        if self.opt.animation:
            self._build_animations()
        cmds.addAttr(self.root, longName='gltfSource', dataType='string')
        cmds.setAttr(self.root + '.gltfSource', doc.path, type='string')
        if self.clip_ranges:
            cmds.addAttr(self.root, longName='gltfClips', dataType='string')
            cmds.setAttr(self.root + '.gltfClips', ';'.join('%s:%g:%g' % c for c in self.clip_ranges), type='string')
        cmds.select(self.root, replace=True)
        return self.root

    # -- nodes -----------------------------------------------------------------
    def _build_node(self, ni, parent):
        node = self.gltf['nodes'][ni]
        name = sanitize_name(node.get('name'), 'node%d' % ni)
        if ni in self.joint_set:
            tname = cmds.createNode('joint', name=name, parent=parent, skipSelect=True)
        else:
            tname = cmds.createNode('transform', name=name, parent=parent, skipSelect=True)
        tname = cmds.ls(tname, long=True)[0]
        self.node_names[ni] = tname
        self._apply_transform(tname, node)
        if 'mesh' in node:
            self.mesh_nodes_pending.add(ni)
        if 'camera' in node and self.opt.cameras:
            self._build_camera(tname, node['camera'])
        light = node.get('extensions', {}).get('KHR_lights_punctual', {}).get('light')
        if light is not None and self.opt.lights:
            self._build_light(tname, light)
        for ci in node.get('children', []):
            self._build_node(ci, tname)

    def _apply_transform(self, tname, node):
        if 'matrix' in node:
            m = om.MMatrix(node['matrix'])
            tm = om.MTransformationMatrix(m)
            t = tm.translation(om.MSpace.kTransform)
            r = tm.rotation(asQuaternion=False)
            sc = tm.scale(om.MSpace.kTransform)
            cmds.setAttr(tname + '.translate', t.x, t.y, t.z)
            cmds.setAttr(tname + '.rotate', math.degrees(r.x), math.degrees(r.y), math.degrees(r.z))
            cmds.setAttr(tname + '.scale', sc[0], sc[1], sc[2])
            return
        t = node.get('translation')
        if t:
            cmds.setAttr(tname + '.translate', t[0], t[1], t[2])
        q = node.get('rotation')
        if q:
            e = om.MQuaternion(q[0], q[1], q[2], q[3]).asEulerRotation()
            cmds.setAttr(tname + '.rotate', math.degrees(e.x), math.degrees(e.y), math.degrees(e.z))
        s = node.get('scale')
        if s:
            cmds.setAttr(tname + '.scale', s[0], s[1], s[2])

    # -- cameras / lights --------------------------------------------------------
    def _build_camera(self, tname, ci):
        cam = self.gltf['cameras'][ci]
        shape = cmds.createNode('camera', name=tname + 'Shape', parent=tname, skipSelect=True)
        if cam.get('type') == 'orthographic':
            o = cam.get('orthographic', {})
            cmds.setAttr(shape + '.orthographic', True)
            cmds.setAttr(shape + '.orthographicWidth', 2.0 * o.get('xmag', 1.0))
            near, far = o.get('znear', 0.1), o.get('zfar', 1000.0)
        else:
            p = cam.get('perspective', {})
            yfov = p.get('yfov', math.radians(45.0))
            aspect = p.get('aspectRatio', cmds.getAttr(shape + '.horizontalFilmAperture') /
                           cmds.getAttr(shape + '.verticalFilmAperture'))
            vfa = cmds.getAttr(shape + '.verticalFilmAperture')
            cmds.setAttr(shape + '.horizontalFilmAperture', vfa * aspect)
            focal = (vfa * 25.4 / 2.0) / math.tan(yfov / 2.0)
            cmds.setAttr(shape + '.focalLength', focal)
            near, far = p.get('znear', 0.1), p.get('zfar', 1000.0)
        cmds.setAttr(shape + '.nearClipPlane', max(near * self.opt.scale, 0.001))
        cmds.setAttr(shape + '.farClipPlane', far * self.opt.scale)

    def _build_light(self, tname, li):
        light = self.gltf['extensions']['KHR_lights_punctual']['lights'][li]
        ltype = light.get('type', 'point')
        node_type = {'directional': 'directionalLight', 'point': 'pointLight', 'spot': 'spotLight'}.get(ltype)
        if not node_type:
            return
        shape = cmds.createNode(node_type, name=tname + 'Shape', parent=tname, skipSelect=True)
        c = light.get('color', [1, 1, 1])
        cmds.setAttr(shape + '.color', c[0], c[1], c[2])
        cmds.setAttr(shape + '.intensity', light.get('intensity', 1.0))
        if ltype == 'spot':
            spot = light.get('spot', {})
            outer = spot.get('outerConeAngle', math.pi / 4.0)
            inner = spot.get('innerConeAngle', 0.0)
            cmds.setAttr(shape + '.coneAngle', math.degrees(outer * 2.0))
            cmds.setAttr(shape + '.penumbraAngle', -math.degrees(outer - inner))
        if ltype in ('point', 'spot') and 'range' in light:
            cmds.setAttr(shape + '.decayRate', 2)

    # -- meshes ----------------------------------------------------------------
    def _build_mesh_for_node(self, ni):
        node = self.gltf['nodes'][ni]
        mi = node['mesh']
        tname = self.node_names[ni]
        skin = node.get('skin') if self.opt.skins else None
        instancing = node.get('extensions', {}).get('EXT_mesh_gpu_instancing')

        # a shared, unskinned mesh becomes a Maya instance of the first copy
        if skin is None and mi in self.mesh_cache:
            shape, md = self.mesh_cache[mi]
            cmds.parent(shape, tname, add=True, shape=True)
            self.mesh_nodes[ni] = (tname, shape, None)
            return

        md = build_mesh_data(self.doc, mi, want_skin=skin is not None, want_morphs=self.opt.morphs)
        if not md.positions:
            self.doc.warn('Mesh %d produced no geometry' % mi)
            return
        mesh = self.gltf['meshes'][mi]
        mesh_name = sanitize_name(mesh.get('name'), tname) + 'Shape'

        if skin is not None:
            # skinned geometry ignores its node transform; it lives in skeleton space under the root
            holder = cmds.createNode('transform', name=sanitize_name(node.get('name'), 'skinnedMesh%d' % ni) + '_skin',
                                     parent=self.root, skipSelect=True)
            holder = cmds.ls(holder, long=True)[0]
            shape = self._create_maya_mesh(md, holder, mesh_name)
        else:
            shape = self._create_maya_mesh(md, tname, mesh_name)
            self.mesh_cache[mi] = (shape, md)
        holder_t = cmds.listRelatives(shape, parent=True, fullPath=True)[0]

        if self.opt.materials:
            self._assign_materials(shape, md)
        else:
            cmds.sets(shape, edit=True, forceElement='initialShadingGroup')

        if self.opt.merge_vertices and skin is None and not md.targets:
            cmds.polyMergeVertex(holder_t, distance=1e-5, alwaysMergeTwoVertices=False, constructionHistory=False)

        bs = None
        if md.targets:
            bs = self._build_blendshape(holder_t, shape, md, mesh, node)
        if skin is not None:
            self._build_skin(holder_t, shape, skin, md)
        self.mesh_nodes[ni] = (holder_t, shape, bs)

        if instancing:
            self._build_instances(tname, shape, instancing)

    def _create_maya_mesh(self, md, parent, name):
        nvert = len(md.positions)
        points = om.MPointArray([om.MPoint(p[0], p[1], p[2]) for p in md.positions])
        nfaces = len(md.triangles) // 3
        counts = om.MIntArray([3] * nfaces)
        connects = om.MIntArray(md.triangles)
        us = om.MFloatArray([uv[0] for uv in md.uv0])
        vs = om.MFloatArray([1.0 - uv[1] for uv in md.uv0])
        fn = om.MFnMesh()
        parent_obj = _mobject(parent)
        result = fn.create(points, counts, connects, us, vs, parent_obj)
        shape_obj = result
        if not result.hasFn(om.MFn.kMesh):
            shape_obj = om.MFnDagNode(result).child(0)
        fn.setObject(shape_obj)
        om.MFnDependencyNode(shape_obj).setName(name)
        shape = om.MFnDagNode(shape_obj).fullPathName()
        fn.assignUVs(counts, connects)
        if md.has_uv1 and md.uv1:
            set_name = fn.createUVSet('uvSet1')
            fn.setUVs(om.MFloatArray([uv[0] for uv in md.uv1]),
                      om.MFloatArray([1.0 - uv[1] for uv in md.uv1]), set_name)
            fn.assignUVs(counts, connects, set_name)
            fn.setCurrentUVSetName('map1')
        if self.opt.normals and md.has_normals and md.normals and len(md.normals) == nvert:
            normals = om.MVectorArray([om.MVector(n[0], n[1], n[2]) for n in md.normals])
            fn.setVertexNormals(normals, om.MIntArray(list(range(nvert))))
        if md.has_colors and md.colors:
            cs = fn.createColorSet('colorSet1', False)
            fn.setCurrentColorSetName(cs)
            colors = om.MColorArray([om.MColor((c[0], c[1], c[2], c[3])) for c in md.colors])
            fn.setVertexColors(colors, om.MIntArray(list(range(nvert))))
            cmds.setAttr(shape + '.displayColors', 1)
        fn.updateSurface()
        return shape

    def _build_instances(self, tname, shape, ext):
        attrs = ext.get('attributes', {})
        t = self.doc.accessor(attrs['TRANSLATION']) if 'TRANSLATION' in attrs else None
        r = self.doc.accessor(attrs['ROTATION']) if 'ROTATION' in attrs else None
        s = self.doc.accessor(attrs['SCALE']) if 'SCALE' in attrs else None
        count = len(t or r or s or [])
        if count == 0:
            return
        # the shape currently sits directly under the node; move it onto per-instance children
        group = cmds.createNode('transform', name=tname + '_instances', parent=tname, skipSelect=True)
        first = None
        for i in range(count):
            inst = cmds.createNode('transform', name=tname + '_inst%d' % i, parent=group, skipSelect=True)
            if t:
                cmds.setAttr(inst + '.translate', *t[i])
            if r:
                e = om.MQuaternion(*r[i]).asEulerRotation()
                cmds.setAttr(inst + '.rotate', math.degrees(e.x), math.degrees(e.y), math.degrees(e.z))
            if s:
                cmds.setAttr(inst + '.scale', *s[i])
            if first is None:
                cmds.parent(shape, inst, shape=True, relative=True)
                shape = cmds.listRelatives(inst, shapes=True, fullPath=True)[0]
                first = inst
            else:
                cmds.parent(shape, inst, add=True, shape=True)

    # -- materials ---------------------------------------------------------------
    def _assign_materials(self, shape, md):
        cmds.sets(shape, edit=True, forceElement='initialShadingGroup')
        single = len(md.prim_faces) == 1
        for mat_idx, first, count in md.prim_faces:
            if mat_idx is None:
                continue
            sg, tex_links = self._material(mat_idx)
            target = shape if single else '%s.f[%d:%d]' % (shape, first, first + count - 1)
            cmds.sets(target, edit=True, forceElement=sg)
            for file_node, texcoord in tex_links:
                if texcoord > 0:
                    try:
                        cmds.uvLink(uvSet='%s.uvSet[%d].uvSetName' % (shape, texcoord), texture=file_node)
                    except RuntimeError:
                        pass

    def _material(self, mi):
        if mi in self.materials:
            return self.materials[mi]
        mat = self.gltf['materials'][mi]
        name = sanitize_name(mat.get('name'), 'material%d' % mi)
        ss = cmds.shadingNode('standardSurface', asShader=True, name=name)
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=ss + 'SG')
        cmds.connectAttr(ss + '.outColor', sg + '.surfaceShader', force=True)
        links = []
        exts = mat.get('extensions', {})
        pbr = mat.get('pbrMetallicRoughness', {})
        alpha_mode = mat.get('alphaMode', 'OPAQUE')

        base = pbr.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0])
        if 'baseColorTexture' in pbr:
            f = self._file_node(pbr['baseColorTexture'], name + '_baseColor', srgb=True, links=links)
            cmds.setAttr(f + '.colorGain', base[0], base[1], base[2])
            cmds.connectAttr(f + '.outColor', ss + '.baseColor', force=True)
            if alpha_mode != 'OPAQUE':
                cmds.setAttr(f + '.alphaGain', base[3])
                for ch in 'RGB':
                    cmds.connectAttr(f + '.outAlpha', ss + '.opacity' + ch, force=True)
        else:
            cmds.setAttr(ss + '.baseColor', base[0], base[1], base[2])
            if alpha_mode != 'OPAQUE' and base[3] < 1.0:
                cmds.setAttr(ss + '.opacity', base[3], base[3], base[3])

        metallic = pbr.get('metallicFactor', 1.0)
        rough = pbr.get('roughnessFactor', 1.0)
        if 'metallicRoughnessTexture' in pbr:
            f = self._file_node(pbr['metallicRoughnessTexture'], name + '_metalRough', srgb=False, links=links)
            cmds.setAttr(f + '.colorGain', 1.0, rough, metallic)
            cmds.connectAttr(f + '.outColorB', ss + '.metalness', force=True)
            cmds.connectAttr(f + '.outColorG', ss + '.specularRoughness', force=True)
        else:
            cmds.setAttr(ss + '.metalness', metallic)
            cmds.setAttr(ss + '.specularRoughness', rough)

        if 'normalTexture' in mat:
            f = self._file_node(mat['normalTexture'], name + '_normal', srgb=False, links=links)
            bump = cmds.shadingNode('bump2d', asUtility=True, name=name + '_bump')
            cmds.setAttr(bump + '.bumpInterp', 1)  # tangent space normals
            cmds.setAttr(bump + '.bumpDepth', mat['normalTexture'].get('scale', 1.0))
            cmds.setAttr(f + '.alphaIsLuminance', True)
            cmds.connectAttr(f + '.outAlpha', bump + '.bumpValue', force=True)
            cmds.connectAttr(bump + '.outNormal', ss + '.normalCamera', force=True)

        strength = exts.get('KHR_materials_emissive_strength', {}).get('emissiveStrength', 1.0)
        if 'emissiveTexture' in mat:
            ef = mat.get('emissiveFactor', [1.0, 1.0, 1.0])
            f = self._file_node(mat['emissiveTexture'], name + '_emissive', srgb=True, links=links)
            cmds.setAttr(f + '.colorGain', ef[0], ef[1], ef[2])
            cmds.connectAttr(f + '.outColor', ss + '.emissionColor', force=True)
            cmds.setAttr(ss + '.emission', strength)
        else:
            ef = mat.get('emissiveFactor', [0.0, 0.0, 0.0])
            if any(c > 0.0 for c in ef):
                cmds.setAttr(ss + '.emissionColor', ef[0], ef[1], ef[2])
                cmds.setAttr(ss + '.emission', strength)

        if 'occlusionTexture' in mat:
            # standardSurface has no AO input; keep the file node around, unconnected, for the user
            f = self._file_node(mat['occlusionTexture'], name + '_occlusion', srgb=False, links=links)
            cmds.addAttr(ss, longName='gltfOcclusionTexture', attributeType='message')
            cmds.connectAttr(f + '.message', ss + '.gltfOcclusionTexture', force=True)

        if 'KHR_materials_transmission' in exts:
            cmds.setAttr(ss + '.transmission', exts['KHR_materials_transmission'].get('transmissionFactor', 0.0))
        if 'KHR_materials_ior' in exts:
            cmds.setAttr(ss + '.specularIOR', exts['KHR_materials_ior'].get('ior', 1.5))
        if 'KHR_materials_clearcoat' in exts:
            cc = exts['KHR_materials_clearcoat']
            cmds.setAttr(ss + '.coat', cc.get('clearcoatFactor', 0.0))
            cmds.setAttr(ss + '.coatRoughness', cc.get('clearcoatRoughnessFactor', 0.0))
        if 'KHR_materials_specular' in exts:
            cmds.setAttr(ss + '.specular', exts['KHR_materials_specular'].get('specularFactor', 1.0))
        if 'KHR_materials_sheen' in exts:
            sh = exts['KHR_materials_sheen']
            cmds.setAttr(ss + '.sheen', 1.0)
            c = sh.get('sheenColorFactor', [0, 0, 0])
            cmds.setAttr(ss + '.sheenColor', c[0], c[1], c[2])
            cmds.setAttr(ss + '.sheenRoughness', sh.get('sheenRoughnessFactor', 0.0))

        self.materials[mi] = (sg, links)
        return self.materials[mi]

    def _file_node(self, tex_info, name, srgb, links):
        tex = self.gltf['textures'][tex_info['index']]
        source = tex.get('source')
        if source is None:
            for ext_name in ('EXT_texture_webp', 'KHR_texture_basisu'):
                source = tex.get('extensions', {}).get(ext_name, {}).get('source')
                if source is not None:
                    break
        f = cmds.shadingNode('file', asTexture=True, isColorManaged=True, name=name)
        p2d = cmds.shadingNode('place2dTexture', asUtility=True, name=name + '_place2d')
        for attr in PLACE2D_CONNECTIONS:
            cmds.connectAttr('%s.%s' % (p2d, attr), '%s.%s' % (f, attr), force=True)
        cmds.connectAttr(p2d + '.outUV', f + '.uvCoord', force=True)
        cmds.connectAttr(p2d + '.outUvFilterSize', f + '.uvFilterSize', force=True)
        if source is not None:
            try:
                path = self.doc.image_path(source, self.textures_dir)
                cmds.setAttr(f + '.fileTextureName', path, type='string')
            except Exception as exc:  # missing texture file, keep going
                self.doc.warn('Texture %s: %s' % (name, exc))
        try:
            cmds.setAttr(f + '.ignoreColorSpaceFileRules', True)
            cmds.setAttr(f + '.colorSpace', 'sRGB' if srgb else 'Raw', type='string')
        except RuntimeError:
            pass
        if 'sampler' in tex:
            smp = self.gltf['samplers'][tex['sampler']]
            for axis, key in (('U', 'wrapS'), ('V', 'wrapT')):
                mode = smp.get(key, 10497)
                cmds.setAttr('%s.wrap%s' % (p2d, axis), mode != 33071)
                cmds.setAttr('%s.mirror%s' % (p2d, axis), mode == 33648)
        tt = tex_info.get('extensions', {}).get('KHR_texture_transform')
        if tt:
            sc = tt.get('scale', [1.0, 1.0])
            off = tt.get('offset', [0.0, 0.0])
            rot = tt.get('rotation', 0.0)
            cmds.setAttr(p2d + '.repeatUV', sc[0], sc[1])
            # glTF v runs downwards; Maya v upwards (we flipped v = 1 - v on import)
            cmds.setAttr(p2d + '.offset', off[0], 1.0 - sc[1] - off[1])
            cmds.setAttr(p2d + '.rotateUV', -math.degrees(rot))
        texcoord = tex_info.get('texCoord', 0)
        if tt and 'texCoord' in tt:
            texcoord = tt['texCoord']
        links.append((f, texcoord))
        return f

    # -- morph targets -------------------------------------------------------------
    def _build_blendshape(self, transform, shape, md, mesh, node):
        nfaces = len(md.triangles) // 3
        counts = om.MIntArray([3] * nfaces)
        connects = om.MIntArray(md.triangles)
        target_transforms = []
        for t, deltas in enumerate(md.targets):
            pts = om.MPointArray([om.MPoint(p[0] + d[0], p[1] + d[1], p[2] + d[2])
                                  for p, d in zip(md.positions, deltas)])
            fn = om.MFnMesh()
            tobj = fn.create(pts, counts, connects)
            om.MFnDependencyNode(tobj).setName(md.target_names[t])
            target_transforms.append(om.MFnDagNode(tobj).fullPathName())
        bs = cmds.blendShape(*(target_transforms + [transform]), frontOfChain=True,
                             name=sanitize_name(mesh.get('name'), 'mesh') + '_blendShape')[0]
        cmds.delete(target_transforms)
        weights = node.get('weights') or mesh.get('weights') or []
        for i, w in enumerate(weights[:len(md.targets)]):
            cmds.setAttr('%s.weight[%d]' % (bs, i), w)
        return bs

    # -- skinning --------------------------------------------------------------------
    def _build_skin(self, transform, shape, si, md):
        skin = self.gltf['skins'][si]
        joint_indices = skin['joints']
        joints = [self.node_names[j] for j in joint_indices if j in self.node_names]
        if len(joints) != len(joint_indices):
            self.doc.warn('Skin %d references joints outside the imported scene' % si)
        if not joints or md.joints is None:
            return
        ibms = self.doc.accessor(skin['inverseBindMatrices']) if 'inverseBindMatrices' in skin else None
        sc = cmds.skinCluster(*(joints + [transform]), toSelectedBones=True, bindMethod=0, skinMethod=0,
                              normalizeWeights=0, obeyMaxInfluences=False, maximumInfluences=8,
                              removeUnusedInfluence=False,
                              name=sanitize_name(skin.get('name'), 'skin%d' % si) + '_skinCluster')[0]
        fn = oma.MFnSkinCluster(_mobject(sc))
        root_inv = om.MMatrix(cmds.getAttr(self.root + '.worldMatrix[0]')).inverse()
        logical = []
        for j, jname in enumerate(joints):
            idx = fn.indexForInfluenceObject(_dagpath(jname))
            logical.append(idx)
            ibm = om.MMatrix(ibms[j]) if ibms else om.MMatrix.kIdentity
            # bindPreMatrix = inverse(joint world matrix at bind) = inv(root) * ibm  (row-vector maths)
            pre = root_inv * ibm
            cmds.setAttr('%s.bindPreMatrix[%d]' % (sc, idx), *[pre[i] for i in range(16)], type='matrix')
        # skinned geometry is in skeleton space: its holder transform sits at the root
        geom = om.MMatrix(cmds.getAttr(self.root + '.worldMatrix[0]'))
        cmds.setAttr(sc + '.geomMatrix', *[geom[i] for i in range(16)], type='matrix')

        nvert = len(md.positions)
        ninf = len(joints)
        order = sorted(range(ninf), key=lambda k: logical[k])
        pos_of = {logical[k]: p for p, k in enumerate(order)}
        values = [0.0] * (nvert * ninf)
        for v, wmap in enumerate(md.joints):
            total = sum(wmap.values()) or 1.0
            row = v * ninf
            for j, w in wmap.items():
                if j < ninf:
                    values[row + pos_of[logical[j]]] += w / total
        comp_fn = om.MFnSingleIndexedComponent()
        comp = comp_fn.create(om.MFn.kMeshVertComponent)
        comp_fn.addElements(om.MIntArray(list(range(nvert))))
        shape_path = _dagpath(shape)
        fn.setWeights(shape_path, comp, om.MIntArray(sorted(logical)), om.MDoubleArray(values), False)
        cmds.setAttr(sc + '.normalizeWeights', 1)
        cmds.setAttr(sc + '.maintainMaxInfluences', 0)
        cmds.skinPercent(sc, shape, normalize=True)

    # -- animation -----------------------------------------------------------------------
    def _build_animations(self):
        anims = self.gltf.get('animations', [])
        if not anims:
            return
        if self.opt.clip >= 0:
            selected = [(self.opt.clip, anims[self.opt.clip])] if self.opt.clip < len(anims) else []
        else:
            selected = list(enumerate(anims))
        offset = 0.0
        overall_min, overall_max = None, None
        for ai, anim in selected:
            duration = 0.0
            for smp in anim.get('samplers', []):
                times = self.doc.accessor(smp['input'])
                if times:
                    duration = max(duration, float(times[-1]))
            start_frame = offset
            for ch in anim.get('channels', []):
                try:
                    self._build_channel(anim, ch, offset)
                except Exception as exc:
                    self.doc.warn('Animation %d channel failed: %s' % (ai, exc))
            end_frame = offset + duration * self.fps
            name = sanitize_name(anim.get('name'), 'clip%d' % ai)
            self.clip_ranges.append((name, round(start_frame, 3), round(end_frame, 3)))
            overall_min = start_frame if overall_min is None else min(overall_min, start_frame)
            overall_max = end_frame if overall_max is None else max(overall_max, end_frame)
            # leave a gap of one second between concatenated clips
            offset = math.ceil(end_frame + self.fps)
        if overall_min is not None:
            cmds.playbackOptions(minTime=overall_min, maxTime=max(overall_max, overall_min + 1),
                                 animationStartTime=overall_min, animationEndTime=max(overall_max, overall_min + 1))
            cmds.currentTime(overall_min)

    def _build_channel(self, anim, ch, offset):
        target = ch['target']
        ni = target.get('node')
        path = target['path']
        if ni is None or ni not in self.node_names:
            return
        smp = anim['samplers'][ch['sampler']]
        times = self.doc.accessor(smp['input'])
        values = self.doc.accessor(smp['output'])
        interp = smp.get('interpolation', 'LINEAR')
        if interp == 'CUBICSPLINE':
            values = values[1::3]
        frames = [offset + float(t) * self.fps for t in times]
        if not frames:
            return
        tangent = {'STEP': oma.MFnAnimCurve.kTangentStep,
                   'LINEAR': oma.MFnAnimCurve.kTangentLinear}.get(interp, oma.MFnAnimCurve.kTangentSmooth)
        node = self.node_names[ni]
        if path == 'translation':
            for k, attr in enumerate(('translateX', 'translateY', 'translateZ')):
                self._set_curve(node, attr, frames, [v[k] for v in values], oma.MFnAnimCurve.kAnimCurveTL, tangent)
        elif path == 'scale':
            for k, attr in enumerate(('scaleX', 'scaleY', 'scaleZ')):
                self._set_curve(node, attr, frames, [v[k] for v in values], oma.MFnAnimCurve.kAnimCurveTU, tangent)
        elif path == 'rotation':
            eulers = []
            prev = None
            for q in values:
                mq = om.MQuaternion(q[0], q[1], q[2], q[3])
                if prev is not None and (mq.x * prev.x + mq.y * prev.y + mq.z * prev.z + mq.w * prev.w) < 0:
                    mq = mq.negateIt()
                e = mq.asEulerRotation()
                if eulers:
                    try:
                        e = e.setToClosestCut(eulers[-1])
                    except Exception:
                        pass
                eulers.append(e)
                prev = mq
            curves = []
            for k, attr in enumerate(('rotateX', 'rotateY', 'rotateZ')):
                vals = [(e.x, e.y, e.z)[k] for e in eulers]
                c = self._set_curve(node, attr, frames, vals, oma.MFnAnimCurve.kAnimCurveTA, tangent)
                if c:
                    curves.append(c)
            if curves and len(frames) > 1:
                try:
                    cmds.filterCurve(*curves, filter='euler')
                except RuntimeError:
                    pass
        elif path == 'weights':
            entry = self.mesh_nodes.get(ni)
            if not entry or not entry[2]:
                return
            bs = entry[2]
            ntargets = cmds.blendShape(bs, query=True, weightCount=True)
            if ntargets == 0 or len(values) < ntargets:
                return
            per_frame = len(values) // len(frames)
            for t in range(min(ntargets, per_frame)):
                vals = [values[i * per_frame + t] for i in range(len(frames))]
                self._set_curve(bs, 'weight', frames, vals, oma.MFnAnimCurve.kAnimCurveTU, tangent, element=t)

    def _set_curve(self, node, attr, frames, values, curve_type, tangent, element=None):
        plug = _plug(node, attr)
        if element is not None:
            plug = plug.elementByLogicalIndex(element)
        if plug.isDestination:
            src = plug.source().node()
            if src.hasFn(om.MFn.kAnimCurve):
                fn = oma.MFnAnimCurve(src)
            else:
                return None
        else:
            fn = oma.MFnAnimCurve()
            fn.create(plug, curve_type)
        unit = om.MTime.uiUnit()
        times = om.MTimeArray([om.MTime(f, unit) for f in frames])
        if curve_type == oma.MFnAnimCurve.kAnimCurveTL:
            # anim curves store internal units (cm); scene values were set in UI units
            ui = om.MDistance.uiUnit()
            vals = om.MDoubleArray([om.MDistance(float(v), ui).asCentimeters() for v in values])
        else:
            vals = om.MDoubleArray([float(v) for v in values])          # radians for TA
        fn.addKeys(times, vals, tangent, tangent, False)
        return fn.name()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def import_file(path, options=None):
    """Imports a .gltf/.glb file. Returns the root transform name."""
    if not IN_MAYA:
        raise RuntimeError('import_file() must run inside Maya')
    options = options or ImportOptions.from_optionvars()
    doc = GltfDocument(path)
    importer = GltfImporter(doc, options)
    cmds.undoInfo(openChunk=True)
    try:
        root = importer.run()
    finally:
        cmds.undoInfo(closeChunk=True)
    for w in doc.warnings:
        om.MGlobal.displayWarning('GLTF2Maya: ' + w)
    summary = 'GLTF2Maya: imported %s -> %s (%d nodes, %d meshes, %d materials' % (
        os.path.basename(path), root, len(importer.node_names), len(importer.mesh_nodes), len(importer.materials))
    if importer.clip_ranges:
        summary += ', clips: ' + ', '.join('%s [%g-%g]' % c for c in importer.clip_ranges)
    om.MGlobal.displayInfo(summary + ')')
    return root


# -- MPxCommand ----------------------------------------------------------------

FLAGS = [
    # short, long, type, attr
    ('-f', '-file', 'string', None),
    ('-s', '-scale', 'double', 'scale'),
    ('-m', '-materials', 'bool', 'materials'),
    ('-sk', '-skins', 'bool', 'skins'),
    ('-mt', '-morphTargets', 'bool', 'morphs'),
    ('-a', '-animation', 'bool', 'animation'),
    ('-c', '-cameras', 'bool', 'cameras'),
    ('-l', '-lights', 'bool', 'lights'),
    ('-n', '-normals', 'bool', 'normals'),
    ('-mv', '-mergeVertices', 'bool', 'merge_vertices'),
    ('-cl', '-clip', 'long', 'clip'),
    ('-td', '-texturesDir', 'string', 'textures_dir'),
]

if IN_MAYA:
    class GltfImportCmd(om.MPxCommand):
        kName = 'gltf2maya'

        def __init__(self):
            super(GltfImportCmd, self).__init__()

        @staticmethod
        def creator():
            return GltfImportCmd()

        @staticmethod
        def syntax():
            s = om.MSyntax()
            types = {'string': om.MSyntax.kString, 'double': om.MSyntax.kDouble,
                     'bool': om.MSyntax.kBoolean, 'long': om.MSyntax.kLong}
            for short, long_, kind, _ in FLAGS:
                s.addFlag(short, long_, types[kind])
            s.setObjectType(om.MSyntax.kStringObjects, 0, 1)
            return s

        def isUndoable(self):
            return False

        def doIt(self, args):
            parser = om.MArgParser(self.syntax(), args)
            path = None
            if parser.isFlagSet('-f'):
                path = parser.flagArgumentString('-f', 0)
            else:
                objs = parser.getObjectStrings()
                if objs:
                    path = objs[0]
            if not path:
                raise RuntimeError('GLTF2Maya: -file is required')
            opt = ImportOptions.from_optionvars()
            for short, _, kind, attr in FLAGS:
                if attr and parser.isFlagSet(short):
                    if kind == 'double':
                        setattr(opt, attr, parser.flagArgumentDouble(short, 0))
                    elif kind == 'bool':
                        setattr(opt, attr, parser.flagArgumentBool(short, 0))
                    elif kind == 'long':
                        setattr(opt, attr, parser.flagArgumentInt(short, 0))
                    else:
                        setattr(opt, attr, parser.flagArgumentString(short, 0))
            root = import_file(path, opt)
            self.setResult(root)

    # -- MPxFileTranslator (File > Import) -------------------------------------
    if hasattr(om, 'MPxFileTranslator'):
        class GltfTranslator(om.MPxFileTranslator):
            kName = 'glTF'

            @staticmethod
            def creator():
                return GltfTranslator()

            def haveReadMethod(self):
                return True

            def haveWriteMethod(self):
                return False

            def defaultExtension(self):
                return 'glb'

            def filter(self):
                return '*.glb;*.gltf'

            def identifyFile(self, fileObject, buffer, size):
                ext = os.path.splitext(fileObject.resolvedFullName())[1].lower()
                if ext in ('.glb', '.gltf'):
                    return om.MPxFileTranslator.kIsMyFileType
                return om.MPxFileTranslator.kNotMyFileType

            def reader(self, fileObject, optionString, accessMode):
                opt = ImportOptions.from_optionvars()
                for token in (optionString or '').split(';'):
                    if '=' not in token:
                        continue
                    key, val = token.split('=', 1)
                    for _, long_, kind, attr in FLAGS:
                        if attr and long_[1:] == key.strip():
                            setattr(opt, attr, float(val) if kind == 'double' else
                                    (int(val) if kind in ('long', 'bool') else val))
                import_file(fileObject.resolvedFullName(), opt)
    else:
        GltfTranslator = None


SHELF_NAME = 'glTF'            # the shelf Maya2glTF's installer creates; shared with the exporter
SHELF_BUTTON_LABEL = 'Import'
SHELF_ICON = 'GLTF2Maya.svg'   # Maya 2020+ rasterises SVG at the UI scale; GLTF2Maya.png + _150/_200 are fallbacks
SHELF_COMMAND = 'import GLTF2Maya\nGLTF2Maya.show_ui()'


def ensure_shelf_button():
    """Adds an Import button to the glTF shelf (creating the shelf if needed). Safe to call repeatedly."""
    if not IN_MAYA or om.MGlobal.mayaState() != om.MGlobal.kInteractive:
        return
    try:
        import maya.mel as mel
        top = mel.eval('$tmp = $gShelfTopLevel')
        if not top or not cmds.tabLayout(top, exists=True):
            return
        shelves = cmds.tabLayout(top, query=True, childArray=True) or []
        if SHELF_NAME not in shelves:
            mel.eval('addNewShelfTab "%s"' % SHELF_NAME)
        shelf = top + '|' + SHELF_NAME
        for child in cmds.shelfLayout(shelf, query=True, childArray=True) or []:
            path = shelf + '|' + child
            if cmds.shelfButton(path, exists=True) and cmds.shelfButton(path, query=True, label=True) == SHELF_BUTTON_LABEL \
                    and 'GLTF2Maya' in (cmds.shelfButton(path, query=True, command=True) or ''):
                cmds.shelfButton(path, edit=True, image=SHELF_ICON, image1=SHELF_ICON, command=SHELF_COMMAND,
                                 sourceType='python')
                return
        cmds.shelfButton(parent=shelf, label=SHELF_BUTTON_LABEL, annotation='Import glTF / GLB (GLTF2Maya)',
                         image=SHELF_ICON, image1=SHELF_ICON, style='iconOnly', width=35, height=35,
                         command=SHELF_COMMAND, sourceType='python')
        om.MGlobal.displayInfo('GLTF2Maya: added an Import button to the %s shelf' % SHELF_NAME)
    except Exception as exc:
        om.MGlobal.displayWarning('GLTF2Maya: could not add the shelf button (%s)' % exc)


def initializePlugin(mobject):
    plugin = om.MFnPlugin(mobject, 'JunkYardRobotBoy', __version__, 'Any')
    plugin.registerCommand(GltfImportCmd.kName, GltfImportCmd.creator, GltfImportCmd.syntax)
    if GltfTranslator is not None:
        try:
            plugin.registerFileTranslator(GltfTranslator.kName, None, GltfTranslator.creator, None, None, False)
        except Exception as exc:
            om.MGlobal.displayWarning('GLTF2Maya: file translator not registered (%s); use the gltf2maya command' % exc)
    # shelves are built after auto-loaded plug-ins, so defer until the UI is up
    try:
        cmds.evalDeferred(ensure_shelf_button, lowestPriority=True)
    except Exception:
        pass


def uninitializePlugin(mobject):
    plugin = om.MFnPlugin(mobject)
    plugin.deregisterCommand(GltfImportCmd.kName)
    if GltfTranslator is not None:
        try:
            plugin.deregisterFileTranslator(GltfTranslator.kName)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# minimal options window
# ---------------------------------------------------------------------------

def show_ui():
    """Opens the glTF import options window."""
    if not IN_MAYA:
        raise RuntimeError('show_ui() must run inside Maya')
    if not cmds.pluginInfo('GLTF2Maya', query=True, loaded=True):
        try:
            cmds.loadPlugin('GLTF2Maya.py')
        except RuntimeError:
            cmds.loadPlugin(os.path.abspath(__file__))
    win = 'GLTF2MayaWindow'
    if cmds.window(win, exists=True):
        cmds.deleteUI(win)
    opt = ImportOptions.from_optionvars()
    cmds.window(win, title='GLTF2Maya %s - import glTF / GLB' % __version__, widthHeight=(420, 420), sizeable=True)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=6, columnOffset=('both', 8))
    cmds.separator(height=4, style='none')
    cmds.rowLayout(numberOfColumns=2, adjustableColumn=1, columnAttach=[(1, 'both', 0), (2, 'right', 0)])
    path_field = cmds.textField(placeholderText='model.glb', text=cmds.optionVar(q='GLTF2Maya_lastFile')
                                if cmds.optionVar(exists='GLTF2Maya_lastFile') else '')

    def browse(*_):
        res = cmds.fileDialog2(fileMode=1, fileFilter='glTF files (*.glb *.gltf);;All files (*.*)',
                               caption='Import glTF')
        if res:
            cmds.textField(path_field, edit=True, text=res[0])
    cmds.button(label='Browse...', width=80, command=browse)
    cmds.setParent('..')

    cmds.frameLayout(label='Units', collapsable=False, marginWidth=6, marginHeight=6)
    scale_field = cmds.floatFieldGrp(label='Scale factor', value1=opt.scale, columnWidth2=(120, 80),
                                     annotation='Multiplier applied on the import root. glTF files are in metres.')
    unit_factor = om.MDistance(1.0, om.MDistance.kMeters).asUnits(om.MDistance.uiUnit())

    def metres(*_):
        cmds.floatFieldGrp(scale_field, edit=True, value1=unit_factor)

    def one(*_):
        cmds.floatFieldGrp(scale_field, edit=True, value1=1.0)
    cmds.rowLayout(numberOfColumns=2, columnWidth2=(200, 200))
    cmds.button(label='Metres -> scene units (x%g)' % unit_factor, command=metres)
    cmds.button(label='1:1 (Maya2glTF round-trip)', command=one)
    cmds.setParent('..')
    cmds.setParent('..')

    cmds.frameLayout(label='Import', collapsable=False, marginWidth=6, marginHeight=6)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=2)
    boxes = {}
    for attr, label in (('materials', 'Materials & textures (standardSurface)'),
                        ('normals', 'Vertex normals (locked)'),
                        ('skins', 'Skins (joints + skinCluster)'),
                        ('morphs', 'Morph targets (blendShape)'),
                        ('animation', 'Animation'),
                        ('cameras', 'Cameras'),
                        ('lights', 'Lights (KHR_lights_punctual)'),
                        ('merge_vertices', 'Merge duplicate vertices (static meshes only)')):
        boxes[attr] = cmds.checkBox(label=label, value=getattr(opt, attr))
    clip_field = cmds.intFieldGrp(label='Animation clip index', value1=opt.clip, columnWidth2=(120, 60),
                                  annotation='-1 imports every animation, concatenated with a one second gap')
    cmds.setParent('..')
    cmds.setParent('..')

    def run(*_):
        path = cmds.textField(path_field, query=True, text=True).strip()
        if not path or not os.path.isfile(path):
            cmds.confirmDialog(title='GLTF2Maya', message='Choose a .glb or .gltf file first.', button=['OK'])
            return
        opt.scale = cmds.floatFieldGrp(scale_field, query=True, value1=True)
        for attr, box in boxes.items():
            setattr(opt, attr, cmds.checkBox(box, query=True, value=True))
        opt.clip = cmds.intFieldGrp(clip_field, query=True, value1=True)
        opt.to_optionvars()
        cmds.optionVar(sv=('GLTF2Maya_lastFile', path))
        import_file(path, opt)

    cmds.separator(height=4, style='none')
    cmds.button(label='IMPORT', height=36, backgroundColor=(0.33, 0.52, 0.33), command=run)
    cmds.showWindow(win)
