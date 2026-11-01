# GLTF2Maya

A glTF 2.0 / GLB **importer** plug-in for Autodesk Maya 2026 / 2027 — the missing other half of
[Maya2glTF](https://github.com/JunkYardRobotBoy/Maya2glTF_Update). Pure Python, one file,
no third-party dependencies, installs in seconds.

## What it imports

| glTF | Maya |
|---|---|
| Node hierarchy (TRS or matrix) | transforms; nodes used by a skin become joints |
| Meshes, any number of primitives | one mesh per glTF mesh, faces assigned per primitive material |
| Normals, TEXCOORD_0 / TEXCOORD_1, COLOR_0 | locked vertex normals, `map1` + `uvSet1`, `colorSet1` |
| Meshes shared by several nodes, `EXT_mesh_gpu_instancing` | Maya instances of one shape |
| Materials (metal/rough PBR) | `standardSurface` + `file`/`place2dTexture` networks |
| Base colour, metallic-roughness, normal, emissive, alpha | baseColor, metalness (B) / specularRoughness (G), bump2d tangent normals, emission, opacity |
| `KHR_texture_transform`, sampler wrap modes | place2dTexture repeat/offset/rotate, wrap/mirror |
| `KHR_materials_emissive_strength`, `_transmission`, `_ior`, `_clearcoat`, `_specular`, `_sheen` | matching standardSurface attributes |
| Occlusion texture | file node left unconnected (standardSurface has no AO slot), linked via `gltfOcclusionTexture` |
| Skins | `skinCluster` using the glTF inverse bind matrices, weights copied 1:1 (any influence count) |
| Morph targets (`extras.targetNames`) | `blendShape`, targets named |
| Animations: translation / rotation / scale / weights, STEP / LINEAR / CUBICSPLINE | anim curves; every clip is imported and concatenated with a 1 s gap, ranges stored on the root (`gltfClips`) |
| Cameras (perspective / orthographic) | camera shapes |
| `KHR_lights_punctual` | directional / point / spot lights |
| Embedded textures (GLB or base64) | extracted to `<file>_textures/` next to the file |
| `KHR_mesh_quantization`, sparse accessors | supported |

Not supported: Draco / meshopt compressed primitives (re-export uncompressed), points & lines,
KTX2/Basis textures (Maya can't read them), `KHR_materials_pbrSpecularGlossiness`.

Everything is parented under one root group named after the file. The import scale is applied on
that group, so the imported data itself is untouched — unparent or freeze it if you need to.

## Install

1. Run `install_GLTF2Maya.bat`. It copies `GLTF2Maya.py` to `Documents\maya\<2026|2027>\plug-ins\`
   (the plug-in) and to `Documents\maya\scripts\` (so `import GLTF2Maya` works from Python).
   Doing it by hand: the same two copies.
2. In Maya: **Windows > Settings/Preferences > Plug-in Manager**, tick *Loaded* and *Auto load*
   for `GLTF2Maya.py`.

## Use

Three ways:

- **File > Import…**, choose the `glTF` file type. Uses the options last set in the window below.
- The options window:

  ```python
  import GLTF2Maya; GLTF2Maya.show_ui()
  ```

  Run it from the **Python** tab of the Script Editor (in MEL: `python("import GLTF2Maya; GLTF2Maya.show_ui()");`)
  or put it on a shelf button with the language set to Python. The window has two scale presets: **Metres → scene units**
  (×100 for a centimetre scene, right for GLBs from the web) and **1:1** (right for files written
  by Maya2glTF at its default scale factor).
- The command, for scripts and batch:

  ```python
  cmds.gltf2maya(file=r'C:/assets/fox.glb', scale=100, animation=True, clip=-1)
  ```

  Flags: `-file/-f`, `-scale/-s`, `-materials/-m`, `-skins/-sk`, `-morphTargets/-mt`, `-animation/-a`,
  `-cameras/-c`, `-lights/-l`, `-normals/-n`, `-mergeVertices/-mv`, `-clip/-cl` (index, −1 = all),
  `-texturesDir/-td`. Returns the root transform.

## Notes

- Animation times are converted with the scene's current frame rate; set the fps *before*
  importing. Rotations are converted to Euler and run through Maya's Euler filter.
- `-mergeVertices` welds the duplicated vertices every glTF exporter produces at UV/normal seams,
  but only on static meshes (skins and blend shapes keep their vertex order).
- Z-up scenes get a 90° X rotation on the root group.
- The skinned mesh is created in skeleton space under the root (glTF ignores the mesh node's own
  transform for skinned meshes), in a `<node>_skin` transform.

## Development

`GLTF2Maya.py` is split in two: `GltfDocument` / `build_mesh_data` (pure Python, no Maya) and
`GltfImporter` (Maya API 2.0 + cmds). The parsing half is exercised by `tests/test_parse.py` against
the Khronos sample assets (`python tests/test_parse.py <glTF-Sample-Assets>/Models`); the Maya half
needs a Maya session.

## License

MIT, see `LICENSE`.
