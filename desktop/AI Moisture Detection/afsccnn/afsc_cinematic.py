
from __future__ import annotations
import os
import math
import random
import numpy as np
from typing import List, Tuple, Optional

from PIL import Image
import matplotlib.pyplot as plt

from manim import *

# ----------------------------
# Optional PyTorch for masks
# ----------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ----------------------------
# Global config (tweakable)
# ----------------------------
config.background_color = "#0a0d16"  # deep blue/black sci-fi vibe

# ----------------------------
# Numeric helpers (DCT proxy)
# ----------------------------
def dct2(img: np.ndarray) -> np.ndarray:
    # Using centered FFT magnitude as a DCT-like proxy for visuals
    return np.fft.fftshift(np.fft.fft2(img))

def idct2(spec: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(spec)).real

def norm01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x - x.min()
    return x / (x.max() + eps)

# ----------------------------
# Data loading
# ----------------------------
def load_demo_image(path: str = "demo_input.jpg", size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    if os.path.exists(path):
        img = Image.open(path).convert("L").resize(size)
        return np.asarray(img, dtype=np.float32) / 255.0
    # Synthetic fallback (swirl/rings)
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w/2, h/2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    th = np.arctan2(y - cy, x - cx)
    z = 0.5 + 0.5 * np.sin(0.09 * r + 4.7 * np.sin(3*th))
    return z.astype(np.float32)

# ----------------------------
# Checkpoint handling
# ----------------------------
def try_load_masks_from_checkpoint(path: str = "best_afsc_dct_model.pth") -> Optional[List[np.ndarray]]:
    if not TORCH_AVAILABLE or not os.path.exists(path):
        print("[AFSC] No torch or checkpoint not found; using synthetic masks.")
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
        masks: List[np.ndarray] = []

        def to_np(m):
            if isinstance(m, torch.Tensor):
                m = m.detach().cpu().numpy()
            m = np.array(m)
            if m.ndim >= 3:
                m = m[0]
            return norm01(m)

        if isinstance(ckpt, dict):
            if "mask_history" in ckpt:
                for m in ckpt["mask_history"]:
                    masks.append(to_np(m))
                if masks:
                    print(f"[AFSC] Loaded {len(masks)} masks from mask_history.")
                    return masks
            if "mask_param" in ckpt:
                masks = [to_np(ckpt["mask_param"])]
                print("[AFSC] Loaded single mask_param.")
                return masks
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                for k, v in ckpt["state_dict"].items():
                    if "afsc" in k and "mask_param" in k:
                        masks.append(to_np(v))
                if masks:
                    print(f"[AFSC] Loaded {len(masks)} masks from state_dict.")
                    return masks

        # Loose state_dict
        if isinstance(ckpt, dict):
            for k, v in ckpt.items():
                if isinstance(k, str) and "mask_param" in k:
                    masks.append(to_np(v))
            if masks:
                print(f"[AFSC] Loaded {len(masks)} masks (loose keys).")
                return masks
    except Exception as e:
        print(f"[AFSC] Failed to load checkpoint: {e}")

    print("[AFSC] No masks found in checkpoint; using synthetic masks.")
    return None

def synthesize_mask_evolution(shape: Tuple[int, int], steps: int = 7) -> List[np.ndarray]:
    H, W = shape
    ks = np.linspace(6, min(H, W)//2, steps).astype(int)
    out = []
    for k in ks:
        m = np.zeros((H, W), dtype=np.float32)
        m[:k, :k] = 1.0
        # Soft vignette for style
        yy, xx = np.mgrid[0:H, 0:W]
        rr = np.sqrt((xx - W/2)**2 + (yy - H/2)**2)
        rr = norm01(rr)
        m = norm01(0.9*m + 0.4*(1.0 - rr))
        out.append(m)
    return out

# ----------------------------
# Visual building blocks
# ----------------------------
class NeonText(VGroup):
    def __init__(self, text: str, color: str = "#00e5ff", glow_width: float = 8, **kwargs):
        super().__init__(**kwargs)
        main = Text(text, weight=BOLD, color=color)
        aura = main.copy().set_stroke(color=color, width=glow_width, opacity=0.35).set_fill(opacity=0)
        self.add(aura, main)

class NeonBox(VGroup):
    def __init__(self, w=4.0, h=2.0, color="#09fbd3", glow=10, **kwargs):
        super().__init__(**kwargs)
        base = Rectangle(width=w, height=h, stroke_color=color, stroke_width=3)
        aura = base.copy().set_stroke(color=color, width=glow, opacity=0.35)
        self.add(aura, base)

class ScanLine(Rectangle):
    def __init__(self, width=10.5, height=0.15, color="#00ffc6", **kwargs):
        super().__init__(width=width, height=height, fill_color=color, fill_opacity=0.22,
                         stroke_color=color, stroke_width=1.2, **kwargs)

class ParticleField(VGroup):
    def __init__(self, n=280, bounds=7.5, base_color="#74f7ff", **kwargs):
        super().__init__(**kwargs)
        for _ in range(n):
            p = Dot(
                point=np.array([
                    random.uniform(-bounds, bounds),
                    random.uniform(-bounds, bounds),
                    random.uniform(-1.5, 1.5),
                ]),
                radius=random.uniform(0.01, 0.03),
                color=base_color,
                opacity=random.uniform(0.25, 0.75)
            )
            vx = random.uniform(-0.2, 0.2)
            vy = random.uniform(-0.2, 0.2)
            vz = random.uniform(-0.03, 0.03)
            p.v = np.array([vx, vy, vz])
            self.add(p)
        self.bounds = bounds

    def start(self):
        def _upd(mob, dt):
            for d in mob:
                d.shift(d.v * dt)
                c = d.get_center()
                # Wrap around
                if c[0] > self.bounds:  d.move_to([c[0]-2*self.bounds, c[1], c[2]])
                if c[0] < -self.bounds: d.move_to([c[0]+2*self.bounds, c[1], c[2]])
                if c[1] > self.bounds:  d.move_to([c[0], c[1]-2*self.bounds, c[2]])
                if c[1] < -self.bounds: d.move_to([c[0], c[1]+2*self.bounds, c[2]])
        self.add_updater(_upd)
        return self

class SpectrumBars(VGroup):
    def __init__(self, n=80, span=10.0, color="#00c6ff", **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.bars = VGroup()
        dx = span / n
        for i in range(n):
            r = Rectangle(width=dx*0.7, height=0.12, fill_color=color, fill_opacity=0.85, stroke_width=0)
            r.move_to(LEFT*span/2 + RIGHT*(i*dx) + DOWN*3.2)
            self.bars.add(r)
        self.add(self.bars)
        self.t = 0.0

    def start(self):
        def _upd(mob, dt):
            self.t += dt
            for i, r in enumerate(self.bars):
                h = 0.3 + 1.25 * abs(math.sin(self.t*1.4 + i*0.27))
                r.set_height(h, stretch=True)
        self.add_updater(_upd)
        return self

class GlowCircle(Circle):
    def __init__(self, radius=1.0, color="#00eaff", **kwargs):
        super().__init__(radius=radius, stroke_color=color, stroke_width=2, **kwargs)
        self.add(self.copy().set_stroke(width=14, opacity=0.22, color=color))

class ConnectorArrow(Arrow):
    def __init__(self, start, end, **kwargs):
        super().__init__(start, end, buff=0.2, color="#d0ff58", stroke_width=2.5,
                         max_tip_length_to_length_ratio=0.2, **kwargs)

# ----------------------------
# Scene 1: Intro
# ----------------------------
class IntroScene(Scene):
    def construct(self):
        starfield = ParticleField().start()
        self.add(starfield)

        top = Line(LEFT*8, RIGHT*8, color="#0df", stroke_width=1.4).to_edge(UP)
        bot = Line(LEFT*8, RIGHT*8, color="#0df", stroke_width=1.4).to_edge(DOWN)
        self.play(FadeIn(top, shift=UP*0.4), FadeIn(bot, shift=DOWN*0.4), run_time=0.6)

        title = NeonText("AFSC — Adaptive Frequency Selective Convolution", color="#00e5ff").scale(0.75).to_edge(UP)
        sub   = Text("Cinematic CNN Visualization", color="#a9b4ff").scale(0.5).next_to(title, DOWN, buff=0.18)
        self.play(Write(title), FadeIn(sub, shift=UP*0.2), run_time=0.9)

        ring = GlowCircle(radius=0.4, color="#ff5ce7").move_to(ORIGIN)
        self.play(Create(ring), run_time=0.6)
        self.play(ring.animate.scale(6.0).set_opacity(0), run_time=1.2, rate_func=linear)
        self.remove(ring)
        self.wait(0.6)

# ----------------------------
# Scene 2: Input + HUD
# ----------------------------
class InputScene(Scene):
    def construct(self):
        bars = SpectrumBars().start()
        self.add(bars)

        scan = ScanLine(width=13.5).move_to(UP*3.3)
        self.add(scan)
        self.play(scan.animate.shift(DOWN*6.6), run_time=2.1, rate_func=there_and_back)

        img = load_demo_image(size=(128, 128))
        plt.imsave("cin_in.png", norm01(img), cmap="gray")
        img_m = ImageMobject("cin_in.png").scale(2.2).shift(LEFT*3.6 + DOWN*0.1)
        box   = NeonBox(w=img_m.get_width()+0.4, h=img_m.get_height()+0.4, color="#12fcbc")
        grp   = VGroup(box, img_m)
        self.play(FadeIn(grp, scale=0.9), run_time=0.9)

        breathe = ValueTracker(0.0)
        def glow_upd(mob, dt):
            breathe.set_value(breathe.get_value() + dt)
            w = 8 + 4*math.sin(breathe.get_value()*2.0)
            box.set_stroke(width=w, opacity=0.28)
        box.add_updater(glow_upd)

        lbl = Text("Input", color="#9be7ff").scale(0.5).next_to(grp, UP, buff=0.15)
        self.play(FadeIn(lbl, shift=UP*0.1), run_time=0.6)
        self.wait(1.0)

# ----------------------------
# Scene 3: 3D Spectrum
# ----------------------------
class SpectrumScene3D(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=65*DEGREES, theta=-45*DEGREES, zoom=1.1)

        img = load_demo_image(size=(64, 64))
        spec = np.abs(dct2(img))
        spec_n = norm01(spec)

        # Build a 3D bar surface (spikes)
        ax = ThreeDAxes(x_range=[-32, 31, 16], y_range=[-32, 31, 16], z_range=[0, 1, 0.25],
                        x_length=7, y_length=7, z_length=3)
        self.add(ax)

        H, W = spec_n.shape
        dx = 7 / W
        dy = 7 / H

        bars = VGroup()
        for yi in range(H):
            for xi in range(W):
                v = spec_n[yi, xi]
                if v < 0.06:
                    continue
                h = 0.2 + 2.2*v
                r = Prism(dimensions=[dx*0.85, dy*0.85, h], fill_color="#ff6bc9", fill_opacity=0.9, stroke_width=0)
                # Map to axis coordinates
                x = -3.5 + xi*dx
                y = -3.5 + yi*dy
                z = h/2
                r.move_to(ax.c2p(x, y, z))
                bars.add(r)

        self.play(FadeIn(bars, lag_ratio=0.002), run_time=1.2)
        self.begin_ambient_camera_rotation(rate=0.07)
        self.wait(2.0)
        self.stop_ambient_camera_rotation()

        lbl = Text("DCT Spectrum (3D spikes)", color="#ff85e5").scale(0.45).to_edge(UP)
        self.play(FadeIn(lbl))
        self.wait(0.8)

# ----------------------------
# Scene 4: Mask Evolution
# ----------------------------
class MaskEvolutionScene(Scene):
    def construct(self):
        img = load_demo_image(size=(128, 128))
        spec = np.abs(dct2(img))
        spec_n = norm01(spec)
        plt.imsave("cin_spec.png", spec_n, cmap="magma")
        spec_m = ImageMobject("cin_spec.png").scale(2.2).shift(RIGHT*3.6 + DOWN*0.1)
        spec_box = NeonBox(w=spec_m.get_width()+0.4, h=spec_m.get_height()+0.4, color="#f72585")
        spec_grp = VGroup(spec_box, spec_m)
        self.play(FadeIn(spec_grp), run_time=0.9)

        # Load or synthesize masks
        masks = try_load_masks_from_checkpoint("best_afsc_dct_model.pth")
        if masks is None:
            masks = synthesize_mask_evolution(spec_n.shape, steps=8)

        mask_imgs = []
        recon_imgs = []
        for i, m in enumerate(masks):
            plt.imsave(f"cin_mask_{i}.png", m, cmap="gray")
            mask_img = ImageMobject(f"cin_mask_{i}.png").scale(1.7).shift(LEFT*1.6 + DOWN*2.4)
            mask_imgs.append(mask_img)

            recon = idct2(spec * m)
            recon_n = norm01(recon)
            plt.imsave(f"cin_recon_{i}.png", recon_n, cmap="gray")
            recon_img = ImageMobject(f"cin_recon_{i}.png").scale(1.7).shift(RIGHT*1.6 + DOWN*2.4)
            recon_imgs.append(recon_img)

        lbl_spec = Text("DCT (2D)", color="#ff85e5").scale(0.45).next_to(spec_grp, UP, buff=0.15)
        self.play(FadeIn(lbl_spec, shift=UP*0.1))

        self.play(FadeIn(mask_imgs[0]), FadeIn(recon_imgs[0]), run_time=0.7)
        lbl_mask  = Text("Frequency Mask (evolution)", color="#00ffc6").scale(0.45).next_to(mask_imgs[0], UP, buff=0.15)
        lbl_recon = Text("Reconstruction", color="#ffd166").scale(0.45).next_to(recon_imgs[0], UP, buff=0.15)
        self.play(FadeIn(lbl_mask), FadeIn(lbl_recon), run_time=0.6)

        sweep = ScanLine(width=spec_m.get_width()*0.95, height=0.12, color="#ff2e88").move_to(spec_m.get_center())
        sweep.set_z_index(5)
        self.add(sweep)

        for i in range(1, len(mask_imgs)):
            self.play(
                Transform(mask_imgs[i-1], mask_imgs[i]),
                Transform(recon_imgs[i-1], recon_imgs[i]),
                sweep.animate.shift(UP*0.25).set_opacity(0.35),
                run_time=0.9, rate_func=smooth
            )
            self.play(sweep.animate.shift(DOWN*0.25), run_time=0.35)

        # Shockwave
        wave = GlowCircle(radius=0.4, color="#00e5ff").move_to(spec_m.get_center())
        self.add(wave)
        self.play(wave.animate.scale(5.0).set_opacity(0), run_time=1.2, rate_func=linear)
        self.remove(wave)
        self.wait(0.6)

# ----------------------------
# Scene 5: Reconstruction focus
# ----------------------------
class ReconstructionScene(Scene):
    def construct(self):
        img = load_demo_image(size=(128, 128))
        spec = np.abs(dct2(img))
        spec_n = norm01(spec)
        masks = try_load_masks_from_checkpoint("best_afsc_dct_model.pth")
        if masks is None:
            masks = synthesize_mask_evolution(spec_n.shape, steps=6)
        final_mask = masks[-1]

        recon = idct2(spec * final_mask)
        recon_n = norm01(recon)
        plt.imsave("cin_recon_final.png", recon_n, cmap="gray")
        recon_m = ImageMobject("cin_recon_final.png").scale(2.4)

        box = NeonBox(w=recon_m.get_width()+0.5, h=recon_m.get_height()+0.5, color="#ffd166")
        grp = VGroup(box, recon_m)
        self.play(FadeIn(grp, scale=0.9), run_time=0.9)

        # Slow glow
        t = ValueTracker(0.0)
        def upd(mob, dt):
            t.increment_value(dt)
            box.set_stroke(width=7+4*math.sin(t.get_value()*2.0), opacity=0.28)
        box.add_updater(upd)

        self.play(FadeIn(Text("Final Reconstruction", color="#ffd166").scale(0.55).next_to(grp, UP, buff=0.15)))
        self.wait(1.5)
        # ----------------------------
# Scene 6: Feature tunnel (simulated)
# ----------------------------
class FeatureTunnelScene(Scene):
    def construct(self):
        panels = VGroup()
        n = 14
        for i in range(n):
            col = interpolate_color(BLUE_E, "#ff6bc9", i/(n-1))
            w = 6.0 - i*0.25
            h = 3.4 - i*0.12
            r = Rectangle(width=w, height=h, stroke_color=col, stroke_width=2)
            r.set_fill(color=col, opacity=0.05 + 0.05*(i%2))
            r.shift(OUT*0.05*i)
            panels.add(r)
        panels.move_to(ORIGIN)
        self.play(LaggedStart(*[FadeIn(p, shift=IN*0.6) for p in panels], lag_ratio=0.08), run_time=1.2)

        # Pulse
        for k in range(2):
            self.play(
                LaggedStart(
                    *[p.animate.set_fill(opacity=0.10 + 0.10*math.sin(i*0.6 + k*0.9))
                      for i, p in enumerate(panels)],
                    lag_ratio=0.05
                ),
                run_time=0.9
            )

        # Streaming particles through the tunnel
        stream = ParticleField(n=120, bounds=6.5, base_color="#aee1ff").start()
        self.add(stream)

        lbl = Text("Feature-space Propagation", color="#aee1ff").scale(0.5).to_edge(UP)
        self.play(FadeIn(lbl, shift=UP*0.1), run_time=0.5)

        # Breathing scale for depth
        breath = ValueTracker(0.0)
        def scale_upd(mob, dt):
            breath.increment_value(dt)
            s = 1.0 + 0.03*math.sin(2.3*breath.get_value())
            mob.set_width(6.0*s)
        panels.add_updater(scale_upd)

        self.wait(1.6)
        panels.clear_updaters()
        stream.clear_updaters()
        self.play(FadeOut(stream), FadeOut(panels), FadeOut(lbl), run_time=0.6)

# ----------------------------
# Scene 7: Outro
# ----------------------------
class OutroScene(Scene):
    def construct(self):
        # Concentric glow rings
        rings = VGroup()
        for r in [0.6, 1.0, 1.6, 2.4, 3.2]:
            rings.add(GlowCircle(radius=r, color="#00eaff"))
        rings.set_opacity(0.0)
        self.add(rings)

        title = NeonText("AFSC", color="#00eaff").scale(1.4)
        sub   = Text("Adaptive Frequency Selective Convolution", color="#a9b4ff").scale(0.5).next_to(title, DOWN)
        self.play(FadeIn(title, scale=0.9), FadeIn(sub, shift=UP*0.1), run_time=0.9)

        # Ripple the rings outward
        for i, g in enumerate(rings):
            self.play(g.animate.set_opacity(0.25).scale(1.25).set_opacity(0.0), run_time=0.5)

        # Credits / callouts
        credits = VGroup(
            Text("DCT domain selection • Mask evolution • Reconstruction", color="#9be7ff").scale(0.45),
            Text("Frequency-aware inductive bias for CNNs", color="#ffd166").scale(0.45),
        ).arrange(DOWN, buff=0.15).next_to(sub, DOWN, buff=0.35)

        self.play(FadeIn(credits, shift=UP*0.1), run_time=0.7)
        self.wait(1.2)

        self.play(FadeOut(credits), FadeOut(title), FadeOut(sub), run_time=0.6)
        self.wait(0.2)

# ----------------------------
# One-shot stitched film: AFSC_FullShowcase
# (Recreates the key beats from all scenes into a single timeline.)
# ----------------------------
class AFSC_FullShowcase(Scene):
    def construct(self):
        # ---------------- Intro beat ----------------
        starfield = ParticleField().start()
        self.add(starfield)
        title = NeonText("AFSC — Adaptive Frequency Selective Convolution", color="#00e5ff").scale(0.75).to_edge(UP)
        sub   = Text("Cinematic CNN Visualization", color="#a9b4ff").scale(0.5).next_to(title, DOWN, buff=0.18)
        self.play(Write(title), FadeIn(sub, shift=UP*0.2), run_time=0.8)

        # Ripple effect
        ring = GlowCircle(radius=0.4, color="#ff5ce7").move_to(ORIGIN)
        self.play(Create(ring), run_time=0.5)
        self.play(ring.animate.scale(5.5).set_opacity(0), run_time=1.0, rate_func=linear)
        self.remove(ring)

        # ---------------- Input with HUD ----------------
        bars = SpectrumBars().start()
        self.add(bars)

        scan = ScanLine(width=13.5).move_to(UP*3.3)
        self.add(scan)
        self.play(scan.animate.shift(DOWN*6.6), run_time=1.6, rate_func=there_and_back)

        img = load_demo_image(size=(128, 128))
        plt.imsave("cin_in2.png", norm01(img), cmap="gray")
        img_m = ImageMobject("cin_in2.png").scale(2.1).shift(LEFT*3.6 + DOWN*0.1)
        box   = NeonBox(w=img_m.get_width()+0.4, h=img_m.get_height()+0.4, color="#12fcbc")
        grp   = VGroup(box, img_m)
        self.play(FadeIn(grp, scale=0.9), run_time=0.8)

        breathe = ValueTracker(0.0)
        def glow_upd(mob, dt):
            breathe.increment_value(dt)
            w = 8 + 4*math.sin(breathe.get_value()*2.0)
            box.set_stroke(width=w, opacity=0.28)
        box.add_updater(glow_upd)

        lbl_in = Text("Input", color="#9be7ff").scale(0.5).next_to(grp, UP, buff=0.15)
        self.play(FadeIn(lbl_in, shift=UP*0.1), run_time=0.5)

        # ---------------- DCT (2D heatmap) ----------------
        spec = np.abs(dct2(img))
        spec_n = norm01(spec)
        plt.imsave("cin_spec2.png", spec_n, cmap="magma")
        spec_m = ImageMobject("cin_spec2.png").scale(2.1).shift(RIGHT*3.6 + DOWN*0.1)
        spec_box = NeonBox(w=spec_m.get_width()+0.4, h=spec_m.get_height()+0.4, color="#f72585")
        spec_grp = VGroup(spec_box, spec_m)

        arrow = ConnectorArrow(grp.get_right(), spec_grp.get_left())
        self.play(FadeIn(spec_grp), GrowArrow(arrow), run_time=0.8)
        lbl_spec = Text("DCT Spectrum", color="#ff85e5").scale(0.45).next_to(spec_grp, UP, buff=0.15)
        self.play(FadeIn(lbl_spec), run_time=0.4)

        # ---------------- Mask evolution + recon ----------------
        masks = try_load_masks_from_checkpoint("best_afsc_dct_model.pth")
        if masks is None:
            masks = synthesize_mask_evolution(spec_n.shape, steps=7)

        mask_imgs = []
        recon_imgs = []
        for i, m in enumerate(masks):
            plt.imsave(f"film_mask_{i}.png", m, cmap="gray")
            mask_img = ImageMobject(f"film_mask_{i}.png").scale(1.5).shift(LEFT*1.6 + DOWN*2.4)
            mask_imgs.append(mask_img)

            recon = idct2(spec * m)
            recon_n = norm01(recon)
            plt.imsave(f"film_recon_{i}.png", recon_n, cmap="gray")
            recon_img = ImageMobject(f"film_recon_{i}.png").scale(1.5).shift(RIGHT*1.6 + DOWN*2.4)
            recon_imgs.append(recon_img)

        self.play(FadeIn(mask_imgs[0]), FadeIn(recon_imgs[0]), run_time=0.6)
        lbl_mask  = Text("Mask", color="#00ffc6").scale(0.45).next_to(mask_imgs[0], UP, buff=0.15)
        lbl_recon = Text("Reconstruction", color="#ffd166").scale(0.45).next_to(recon_imgs[0], UP, buff=0.15)
        self.play(FadeIn(lbl_mask), FadeIn(lbl_recon), run_time=0.4)

        sweep = ScanLine(width=spec_m.get_width()*0.95, height=0.12, color="#ff2e88").move_to(spec_m.get_center())
        sweep.set_z_index(5)
        self.add(sweep)

        for i in range(1, len(mask_imgs)):
            self.play(
                Transform(mask_imgs[i-1], mask_imgs[i]),
                Transform(recon_imgs[i-1], recon_imgs[i]),
                sweep.animate.shift(UP*0.25).set_opacity(0.35),
                run_time=0.8, rate_func=smooth
            )
            self.play(sweep.animate.shift(DOWN*0.25), run_time=0.3)

        wave = GlowCircle(radius=0.4, color="#00e5ff").move_to(spec_m.get_center())
        self.add(wave)
        self.play(wave.animate.scale(5.0).set_opacity(0), run_time=1.0, rate_func=linear)
        self.remove(wave)

        # ---------------- Final recon focus ----------------
        final_mask = masks[-1]
        recon = idct2(spec * final_mask)
        recon_n = norm01(recon)
        plt.imsave("film_recon_final.png", recon_n, cmap="gray")
        recon_m = ImageMobject("film_recon_final.png").scale(2.2).shift(DOWN*0.1)

        # Clear top row UI
        self.play(
            FadeOut(lbl_mask), FadeOut(lbl_recon),
            FadeOut(lbl_spec), FadeOut(lbl_in),
            FadeOut(arrow),
            run_time=0.4
        )

        # Bring the final recon center stage
        self.play(
            FadeOut(mask_imgs[-1]),
            FadeOut(recon_imgs[-1]),
            FadeOut(spec_grp),
            FadeOut(grp),
            FadeIn(recon_m, scale=0.9),
            run_time=0.9
        )

        box = NeonBox(w=recon_m.get_width()+0.5, h=recon_m.get_height()+0.5, color="#ffd166")
        self.play(FadeIn(box), run_time=0.4)
        t = ValueTracker(0.0)
        def upd(mob, dt):
            t.increment_value(dt)
            box.set_stroke(width=7+4*math.sin(t.get_value()*2.0), opacity=0.28)
        box.add_updater(upd)
        self.play(FadeIn(Text("Final Reconstruction", color="#ffd166").scale(0.55).next_to(recon_m, UP, buff=0.15)))
        self.wait(1.1)

        # ---------------- Feature tunnel beat ----------------
        panels = VGroup()
        n = 12
        for i in range(n):
            col = interpolate_color(BLUE_E, "#ff6bc9", i/(n-1))
            w = 6.0 - i*0.28
            h = 3.4 - i*0.12
            r = Rectangle(width=w, height=h, stroke_color=col, stroke_width=2)
            r.set_fill(color=col, opacity=0.05 + 0.05*(i%2))
            r.shift(OUT*0.05*i)
            panels.add(r)
        panels.move_to(ORIGIN).set_opacity(0.0)
        self.play(FadeIn(panels, shift=IN*0.6), run_time=0.6)
        self.play(panels.animate.set_opacity(1.0), run_time=0.3)

        stream = ParticleField(n=100, bounds=6.0, base_color="#aee1ff").start()
        self.add(stream)
        self.wait(0.9)
        panels.clear_updaters()
        stream.clear_updaters()

        self.play(FadeOut(stream), FadeOut(panels), FadeOut(box), FadeOut(recon_m), run_time=0.6)

        # ---------------- Outro ----------------
        rings = VGroup()
        for r in [0.6, 1.0, 1.6, 2.4, 3.2]:
            rings.add(GlowCircle(radius=r, color="#00eaff"))
        rings.set_opacity(0.0)
        self.add(rings)

        title2 = NeonText("AFSC", color="#00eaff").scale(1.4)
        sub2   = Text("Adaptive Frequency Selective Convolution", color="#a9b4ff").scale(0.5).next_to(title2, DOWN)
        self.play(FadeIn(title2, scale=0.9), FadeIn(sub2, shift=UP*0.1), run_time=0.8)

        for g in rings:
            self.play(g.animate.set_opacity(0.25).scale(1.25).set_opacity(0.0), run_time=0.45)

        credits = VGroup(
            Text("DCT → Mask → Reconstruction → Feature Flow", color="#9be7ff").scale(0.45),
            Text("Frequency-aware inductive bias for CNNs", color="#ffd166").scale(0.45),
        ).arrange(DOWN, buff=0.15).next_to(sub2, DOWN, buff=0.35)
        self.play(FadeIn(credits, shift=UP*0.1), run_time=0.6)
        self.wait(1.2)

        self.play(FadeOut(credits), FadeOut(title2), FadeOut(sub2), run_time=0.6)
        self.wait(0.2)