"""
Augment genuine PDFs into variants, generate tampered samples, then train.

Usage:
  python3 augment_and_train.py --genuine path/to/genuine.pdf
  python3 augment_and_train.py --tamper  path/to/genuine.pdf
  python3 augment_and_train.py --train
  python3 augment_and_train.py --stats
"""

import argparse
import io
import os
import sys
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), 'training_data')
GENUINE  = os.path.join(DATA_DIR, 'genuine')
TAMPERED = os.path.join(DATA_DIR, 'tampered')
os.makedirs(GENUINE,  exist_ok=True)
os.makedirs(TAMPERED, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────

def _add_noise(img, amount):
    arr   = np.array(img).astype(np.int16)
    noise = np.random.randint(-amount, amount, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def _jpeg_recompress(img, quality):
    buf = io.BytesIO()
    img.convert('RGB').save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def _img_to_pdf(img, out_path):
    """Save a PIL image as a single-page PDF using reportlab."""
    from reportlab.pdfgen import canvas as rl_canvas
    w, h  = img.size
    buf   = io.BytesIO()
    tmp   = f"/tmp/_aug_tmp.jpg"
    img.convert('RGB').save(tmp, 'JPEG', quality=88)
    c = rl_canvas.Canvas(buf, pagesize=(w * 0.75, h * 0.75))
    c.drawImage(tmp, 0, 0, width=w * 0.75, height=h * 0.75)
    c.save()
    os.unlink(tmp)
    with open(out_path, 'wb') as f:
        f.write(buf.getvalue())


# ── Command: --genuine ─────────────────────────────────────────────

def augment_genuine(pdf_path):
    """Turn 1 genuine PDF into 20 augmented variants."""
    from pdf2image import convert_from_bytes

    print(f"Reading: {pdf_path}")
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    images = convert_from_bytes(pdf_bytes, dpi=150)
    img    = images[0]
    pw, ph = img.size

    augmentations = [
        ("orig",       lambda i: i),
        ("bright+",    lambda i: ImageEnhance.Brightness(i).enhance(1.15)),
        ("bright-",    lambda i: ImageEnhance.Brightness(i).enhance(0.87)),
        ("contrast+",  lambda i: ImageEnhance.Contrast(i).enhance(1.20)),
        ("contrast-",  lambda i: ImageEnhance.Contrast(i).enhance(0.82)),
        ("sharp+",     lambda i: ImageEnhance.Sharpness(i).enhance(1.5)),
        ("sharp-",     lambda i: ImageEnhance.Sharpness(i).enhance(0.5)),
        ("blur",       lambda i: i.filter(ImageFilter.GaussianBlur(1))),
        ("noise",      lambda i: _add_noise(i, 8)),
        ("noise2",     lambda i: _add_noise(i, 15)),
        ("crop_t",     lambda i: i.crop((0, int(ph * 0.02), pw, ph))),
        ("crop_b",     lambda i: i.crop((0, 0, pw, int(ph * 0.98)))),
        ("crop_l",     lambda i: i.crop((int(pw * 0.02), 0, pw, ph))),
        ("crop_r",     lambda i: i.crop((0, 0, int(pw * 0.98), ph))),
        ("rotate+",    lambda i: i.rotate(1.5,  expand=False, fillcolor=(240, 240, 240))),
        ("rotate-",    lambda i: i.rotate(-1.5, expand=False, fillcolor=(240, 240, 240))),
        ("jpeg_lo",    lambda i: _jpeg_recompress(i, 65)),
        ("jpeg_mid",   lambda i: _jpeg_recompress(i, 80)),
        ("resize_sm",  lambda i: i.resize((int(pw * 0.85), int(ph * 0.85)), Image.LANCZOS)),
        ("resize_lg",  lambda i: i.resize((int(pw * 1.15), int(ph * 1.15)), Image.LANCZOS)),
    ]

    base  = os.path.splitext(os.path.basename(pdf_path))[0]
    saved = 0
    for aug_name, transform in augmentations:
        try:
            aug_img  = transform(img.copy())
            out_path = os.path.join(GENUINE, f"{base}_{aug_name}.pdf")
            _img_to_pdf(aug_img, out_path)
            saved += 1
            print(f"  ✅ {aug_name}")
        except Exception as e:
            print(f"  ⚠️  {aug_name} failed: {e}")

    print(f"\n✅ Created {saved} genuine variants → training_data/genuine/")


# ── Command: --tamper ──────────────────────────────────────────────

def generate_tampered(pdf_path, num_samples=50):
    """Generate multiple tampered versions from one PDF."""
    from pdf2image import convert_from_bytes

    print(f"🔧 Generating {num_samples} tampered samples from: {pdf_path}")

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    images = convert_from_bytes(pdf_bytes, dpi=150)
    img    = images[0].convert("RGB")
    w, h   = img.size

    tamper_types = ["text_edit", "copy_paste", "blur", "noise", "overlay"]

    for i in range(num_samples):
        tampered     = img.copy()
        draw         = ImageDraw.Draw(tampered)
        tamper_type  = random.choice(tamper_types)

        if tamper_type == "text_edit":
            draw.rectangle((150, 250, 350, 300), fill=(255, 255, 255))
            draw.text((160, 255), str(random.randint(50000, 99999)), fill=(0, 0, 0))

        elif tamper_type == "copy_paste":
            crop = tampered.crop((100, 100, 200, 150))
            tampered.paste(crop, (300, 300))

        elif tamper_type == "blur":
            region = tampered.crop((200, 200, 350, 300))
            region = region.filter(ImageFilter.GaussianBlur(3))
            tampered.paste(region, (200, 200))

        elif tamper_type == "noise":
            arr      = np.array(tampered).astype(np.int16)
            noise    = np.random.randint(-20, 20, arr.shape)
            tampered = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

        elif tamper_type == "overlay":
            draw.rectangle((50, 50, 200, 100), fill=(200, 200, 200))
            draw.text((60, 60), "APPROVED", fill=(0, 0, 0))

        out_path = os.path.join(TAMPERED, f"tampered_{i}.pdf")
        _img_to_pdf(tampered, out_path)
        print(f"  ✅ tampered_{i} ({tamper_type})")

    print(f"\n🔥 Generated {num_samples} tampered samples → training_data/tampered/")


# ── Command: --train ───────────────────────────────────────────────

def train():
    """Train the model on genuine + tampered PDFs."""
    genuine_pdfs  = [os.path.join(GENUINE,  f) for f in os.listdir(GENUINE)  if f.endswith('.pdf')]
    tampered_pdfs = [os.path.join(TAMPERED, f) for f in os.listdir(TAMPERED) if f.endswith('.pdf')]

    print(f"\n📊 Training data:")
    print(f"   Genuine:  {len(genuine_pdfs)}")
    print(f"   Tampered: {len(tampered_pdfs)}")

    if len(genuine_pdfs) < 3 or len(tampered_pdfs) < 1:
        print("❌ Need at least 3 genuine + 1 tampered. Run --genuine and --tamper first.")
        return

    try:
        import cv2
        from pdf2image import convert_from_bytes
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score
        import pickle
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install scikit-learn opencv-python pdf2image")
        return

    def extract_features(pdf_path):
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            images = convert_from_bytes(pdf_bytes, dpi=100)
            img    = images[0]
            rgb    = np.array(img.convert('RGB')).astype(np.float32)
            gray   = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            h, w   = gray.shape
            feats  = []

            # Color R/B ratios (header vs body)
            def rb(zone):
                dark = zone.mean(axis=2) < 110
                if dark.sum() < 20:
                    return 1.0
                return float(zone[:, :, 0][dark].mean() /
                             (zone[:, :, 2][dark].mean() + 1e-6))

            feats += [rb(rgb[:int(h * .28), :]),
                      rb(rgb[int(h * .28):int(h * .80), :int(w * .60)])]
            feats.append(feats[1] - feats[0])

            # Noise per zone
            for y0, y1 in [(.0, .25), (.25, .50), (.50, .75), (.75, 1.)]:
                zone = gray[int(h * y0):int(h * y1), :]
                blur = cv2.GaussianBlur(zone, (5, 5), 0)
                feats.append(float((zone.astype(float) - blur).std()))

            # Sharpness
            body = gray[int(h * .25):int(h * .85), :int(w * .60)]
            feats.append(float(cv2.Laplacian(body.astype(np.float32), cv2.CV_32F).var()))

            # DCT frequency ratios
            P = 64
            ratios = []
            for y in range(0, h - P, P * 2):
                for x in range(0, w // 2 - P, P * 2):
                    tile = gray[y:y + P, x:x + P].astype(np.float32)
                    if tile.mean() > 235:
                        continue
                    dct = cv2.dct(tile)
                    ratios.append(float(
                        np.abs(dct[16:, 16:]).mean() /
                        (np.abs(dct[:8, :8]).mean() + 1e-6)
                    ))
            feats += [
                np.mean(ratios) if ratios else 0,
                np.std(ratios)  if ratios else 0,
                np.max(ratios)  if ratios else 0,
            ]

            # Bimodality
            rd = (gray < 100).sum(axis=1).astype(float)
            tr = rd[rd > w * .01]
            feats.append(
                float(np.percentile(tr, 75) / (np.percentile(tr, 25) + 1e-6))
                if len(tr) >= 4 else 1.0
            )

            # Hologram blank ratio
            holo = gray[:int(h * .45), int(w * .55):]
            B    = 22
            blank = tot = 0
            for y in range(0, holo.shape[0] - B, B):
                for x in range(0, holo.shape[1] - B, B):
                    tile = holo[y:y + B, x:x + B]
                    tot += 1
                    if tile.std() < 12 and 150 < float(tile.mean()) < 235:
                        blank += 1
            feats.append(blank / (tot + 1e-6))

            return np.array(feats, dtype=np.float32)
        except Exception:
            return None

    print("\n🔄 Extracting features...")
    X, y = [], []
    for path in genuine_pdfs:
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(0)
    for path in tampered_pdfs:
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(1)

    X = np.array(X)
    y = np.array(y)
    print(f"   Extracted: {len(X)} samples, {X.shape[1]} features")

    model = Pipeline([
        ('sc',  StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            random_state=42, learning_rate=0.1
        )),
    ])

    if len(X) >= 6:
        cv = min(5, len(X[y == 0]), len(X[y == 1]))
        if cv >= 2:
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            print(f"   CV F1: {scores.mean():.2f} ± {scores.std():.2f}")

    model.fit(X, y)

    model_path = os.path.join(os.path.dirname(__file__), 'detector', 'cnn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model':      model,
            'n_genuine':  len(genuine_pdfs),
            'n_tampered': len(tampered_pdfs),
            'version':    2,
        }, f)

    print(f"\n✅ Model saved → {model_path}")


# ── Command: --stats ───────────────────────────────────────────────

def show_stats():
    genuine_pdfs  = [f for f in os.listdir(GENUINE)  if f.endswith('.pdf')]
    tampered_pdfs = [f for f in os.listdir(TAMPERED) if f.endswith('.pdf')]
    print(f"\n📊 Training Data")
    print(f"   Genuine:  {len(genuine_pdfs):3d} PDFs → training_data/genuine/")
    print(f"   Tampered: {len(tampered_pdfs):3d} PDFs → training_data/tampered/")
    if len(genuine_pdfs) >= 15 and len(tampered_pdfs) >= 3:
        print("   ✅ Ready to train:  python3 augment_and_train.py --train")
    else:
        if len(genuine_pdfs) < 15:
            print("   ⚠️  Need more genuine: python3 augment_and_train.py --genuine your.pdf")
        if len(tampered_pdfs) < 3:
            print("   ⚠️  Need more tampered: python3 augment_and_train.py --tamper your.pdf")


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Augment and train PDF tamper detector")
    p.add_argument('--genuine', help='Path to a genuine PDF to augment into 20 variants')
    p.add_argument('--tamper',  help='Path to a genuine PDF to generate tampered versions from')
    p.add_argument('--train',   action='store_true', help='Train the model')
    p.add_argument('--stats',   action='store_true', help='Show training data stats')
    args = p.parse_args()

    if args.genuine:
        augment_genuine(args.genuine)
    elif args.tamper:
        generate_tampered(args.tamper)
    elif args.train:
        train()
    elif args.stats:
        show_stats()
    else:
        p.print_help()