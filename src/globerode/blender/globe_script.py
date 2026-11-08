"""Blender globe scene for a globerode run (DESIGN §12.3, SPEC §12.1).

This file is a template: globerode writes a copy into each run directory as
``blender_globe.py`` with SETTINGS filled in. It does not import globerode,
so it runs in any Blender 5.0+:

    blender -b -P blender_globe.py -- --render globe.png      # Blender app
    python blender_globe.py --render globe.png                # bpy module
    (or run its contents through the Blender MCP)

Without --render it only builds the scene (and saves it with --save).

Scene: a radius-1 UV sphere displaced by the height EXR (Cycles adaptive
subdivision for renders, a Displace modifier for the viewport), a
hypsometric colour ramp with rivers, a sea sphere, a sun and a camera.
"""

import json
import math
import os
import sys

import bpy

SETTINGS: dict = {}  # globerode replaces this line with the run's settings

HERE = os.path.dirname(os.path.abspath(__file__))


def _path(rel):
    return rel if rel is None or os.path.isabs(rel) else os.path.join(HERE, rel)


def _args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    opts = {"render": None, "save": None, "size": 1024, "samples": 32, "lon": 0.0, "lat": 20.0}
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-")
        if key in opts and i + 1 < len(argv):
            val = argv[i + 1]
            opts[key] = type(opts[key])(val) if opts[key] is not None else val
            i += 2
        else:
            i += 1
    return opts


def _linear(srgb):
    """sRGB display colour -> linear RGBA."""
    out = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*out, 1.0)


def _image(path, colorspace="Non-Color"):
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = colorspace
    return img


def _lonlat_uv(nodes, links):
    """Object coords -> (u, v) equirect coords: u = lon/2pi + 0.5, v = lat/pi + 0.5."""
    tc = nodes.new("ShaderNodeTexCoord")
    norm = nodes.new("ShaderNodeVectorMath")
    norm.operation = "NORMALIZE"
    links.new(tc.outputs["Object"], norm.inputs[0])
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(norm.outputs["Vector"], sep.inputs[0])
    lon = nodes.new("ShaderNodeMath")
    lon.operation = "ARCTAN2"
    links.new(sep.outputs["Y"], lon.inputs[0])
    links.new(sep.outputs["X"], lon.inputs[1])
    u = nodes.new("ShaderNodeMath")
    u.operation = "MULTIPLY_ADD"
    links.new(lon.outputs[0], u.inputs[0])
    u.inputs[1].default_value = 1.0 / (2.0 * math.pi)
    u.inputs[2].default_value = 0.5
    lat = nodes.new("ShaderNodeMath")
    lat.operation = "ARCSINE"
    links.new(sep.outputs["Z"], lat.inputs[0])
    v = nodes.new("ShaderNodeMath")
    v.operation = "MULTIPLY_ADD"
    links.new(lat.outputs[0], v.inputs[0])
    v.inputs[1].default_value = 1.0 / math.pi
    v.inputs[2].default_value = 0.5
    comb = nodes.new("ShaderNodeCombineXYZ")
    links.new(u.outputs[0], comb.inputs["X"])
    links.new(v.outputs[0], comb.inputs["Y"])
    return comb.outputs["Vector"]


def _land_material(s):
    mat = bpy.data.materials.new("globerode_land")
    mat.use_nodes = True
    mat.displacement_method = "DISPLACEMENT"
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.85
    links.new(bsdf.outputs[0], out.inputs["Surface"])
    uv = _lonlat_uv(nodes, links)

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = _image(_path(s["height_exr"]))
    tex.interpolation = "Cubic"
    tex.extension = "REPEAT"
    links.new(uv, tex.inputs["Vector"])

    disp = nodes.new("ShaderNodeDisplacement")
    disp.inputs["Midlevel"].default_value = 0.0
    disp.inputs["Scale"].default_value = s["exaggeration"] / s["radius_m"]
    links.new(tex.outputs["Color"], disp.inputs["Height"])
    links.new(disp.outputs[0], out.inputs["Displacement"])

    rng = nodes.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value = 0.0
    rng.inputs["From Max"].default_value = max(s["max_height_m"], 1.0)
    links.new(tex.outputs["Color"], rng.inputs["Value"])
    ramp = nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    # Display (sRGB) colours, converted to the linear values Blender works in.
    stops = [
        (pos, _linear(rgb))
        for pos, rgb in [
            (0.0, (0.35, 0.55, 0.30)),
            (0.08, (0.55, 0.62, 0.35)),
            (0.25, (0.70, 0.62, 0.42)),
            (0.5, (0.58, 0.45, 0.35)),
            (0.75, (0.88, 0.88, 0.88)),
            (1.0, (1.0, 1.0, 1.0)),
        ]
    ]
    els[0].position, els[0].color = stops[0]
    els[1].position, els[1].color = stops[-1]
    for pos, col in stops[1:-1]:
        e = els.new(pos)
        e.color = col
    links.new(rng.outputs["Result"], ramp.inputs["Fac"])
    color = ramp.outputs["Color"]

    if s.get("rivers_png"):
        riv = nodes.new("ShaderNodeTexImage")
        riv.image = _image(_path(s["rivers_png"]))
        riv.interpolation = "Linear"
        links.new(uv, riv.inputs["Vector"])
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        links.new(riv.outputs["Color"], mix.inputs["Factor"])
        links.new(color, mix.inputs["A"])
        mix.inputs["B"].default_value = _linear((0.25, 0.45, 0.75))
        color = mix.outputs["Result"]
    links.new(color, bsdf.inputs["Base Color"])
    return mat, tex.image


def _sea_material():
    mat = bpy.data.materials.new("globerode_sea")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = _linear((0.10, 0.22, 0.42))
    bsdf.inputs["Roughness"].default_value = 0.25
    return mat


def build_scene(s):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.view_settings.view_transform = "Standard"  # keep the colour ramp's colours
    world = bpy.data.worlds.new("globerode_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.world = world

    bpy.ops.mesh.primitive_uv_sphere_add(segments=256, ring_count=128, radius=1.0)
    globe = bpy.context.active_object
    globe.name = "globe"
    bpy.ops.object.shade_smooth()
    mat, img = _land_material(s)
    globe.data.materials.append(mat)

    # Viewport preview: Displace modifier along normals (not used in renders).
    tex = bpy.data.textures.new("globerode_height", type="IMAGE")
    tex.image = img
    tex.extension = "REPEAT"
    disp = globe.modifiers.new("preview_displace", "DISPLACE")
    disp.texture = tex
    disp.texture_coords = "UV"
    disp.mid_level = 0.0
    disp.strength = s["exaggeration"] / s["radius_m"]
    disp.show_render = False

    # Render: adaptive subdivision + true displacement in the material.
    sub = globe.modifiers.new("adaptive", "SUBSURF")
    sub.subdivision_type = "SIMPLE"
    sub.use_adaptive_subdivision = True
    sub.adaptive_space = "PIXEL"
    sub.adaptive_pixel_size = 1.0
    sub.levels = 0
    sub.show_viewport = False

    bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=1.0)
    sea = bpy.context.active_object
    sea.name = "sea"
    bpy.ops.object.shade_smooth()
    sea.data.materials.append(_sea_material())

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 2.5
    sun_obj = bpy.data.objects.new("sun", sun)
    scene.collection.objects.link(sun_obj)

    cam = bpy.data.cameras.new("camera")
    cam.type = "ORTHO"
    cam.ortho_scale = 2.5  # the globe (diameter 2) fills 80% of the frame
    cam_obj = bpy.data.objects.new("camera", cam)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj, sun_obj


def aim(cam_obj, sun_obj, lon_deg, lat_deg):
    lo, la = math.radians(lon_deg), math.radians(lat_deg)
    d = (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))
    cam_obj.location = tuple(4.0 * x for x in d)
    look = cam_obj.location.to_track_quat("Z", "Y")  # camera looks along -Z
    cam_obj.rotation_euler = look.to_euler()
    sun_obj.rotation_euler = (
        (cam_obj.location + type(cam_obj.location)((0.0, 0.0, 1.5)))
        .to_track_quat("Z", "Y")
        .to_euler()
    )


def render(path, size, samples):
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.cycles.samples = samples
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    opts = _args()
    cam, sun = build_scene(SETTINGS)
    aim(cam, sun, opts["lon"], opts["lat"])
    if opts["save"]:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(opts["save"]))
    if opts["render"]:
        render(os.path.abspath(opts["render"]), opts["size"], opts["samples"])
    print(json.dumps({"ok": True, "render": opts["render"], "save": opts["save"]}))


if __name__ == "__main__":
    main()
