import sys, glob, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import GLTF2Maya as g
# usage: python tests/test_parse.py <path to a clone of KhronosGroup/glTF-Sample-Assets>/Models
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'samples')
files = sorted(glob.glob(os.path.join(root, '*', 'glTF*', '*.gl*')))
for f in files:
    if 'Draco' in f: continue
    try:
        doc = g.GltfDocument(f)
        n_tri = 0; n_v = 0; targets = 0; skinned = 0; mats=set()
        for mi in range(len(doc.gltf.get('meshes', []))):
            md = g.build_mesh_data(doc, mi)
            n_tri += len(md.triangles)//3; n_v += len(md.positions)
            targets += len(md.targets); skinned += md.joints is not None
            assert len(md.uv0) == len(md.positions)
            if md.normals is not None: assert len(md.normals) == len(md.positions)
            for t in md.targets: assert len(t) == len(md.positions)
            if md.joints is not None:
                assert len(md.joints) == len(md.positions)
                assert all(abs(sum(w.values())-1) < 1e-3 for w in md.joints if w), 'weights not normalised'
            for m,a,c in md.prim_faces: mats.add(m)
            assert max(md.triangles) < len(md.positions)
        anims = doc.gltf.get('animations', [])
        keys = 0
        for a in anims:
            for s in a['samplers']:
                t = doc.accessor(s['input']); v = doc.accessor(s['output'])
                mult = 3 if s.get('interpolation')=='CUBICSPLINE' else 1
                if a['channels'][0]['target']['path'] != 'weights':
                    pass
                keys += len(t)
        imgs = []
        for i in range(len(doc.gltf.get('images', []))):
            p = doc.image_path(i, os.path.join(root, '_textures_test', '')+doc.name)
            imgs.append(os.path.getsize(p))
        for si, skin in enumerate(doc.gltf.get('skins', [])):
            if 'inverseBindMatrices' in skin:
                ibm = doc.accessor(skin['inverseBindMatrices']); assert len(ibm)==len(skin['joints']) and len(ibm[0])==16
        print('%-45s nodes=%d roots=%s tris=%d verts=%d mats=%d targets=%d skinned=%d anims=%d keys=%d imgs=%s warn=%s' % (
            os.path.relpath(f, root), len(doc.nodes()), doc.scene_roots(), n_tri, n_v, len(mats), targets, skinned, len(anims), keys, imgs, doc.warnings))
    except Exception as e:
        import traceback; traceback.print_exc()
        print('FAIL', f, e)
