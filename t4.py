import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


def display_detected_fragments(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image could not be loaded. Check your file path.")
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = (h * w) * 0.005
    fragments = [c for c in contours if cv2.contourArea(c) > min_area]

    temporary_storage = []
    for idx, contour in enumerate(fragments):
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        piece = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        x, y, pw, ph = cv2.boundingRect(contour)
        cropped_piece = piece[y : y + ph, x : x + pw]

        cropped_bgr = cv2.cvtColor(cropped_piece, cv2.COLOR_RGB2BGR)

        temporary_storage.append(
            {"img": cropped_bgr, "orig_x": x, "orig_y": y}
        )

    return temporary_storage, w


def get_mask(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return (gray > 15).astype(np.uint8) * 255


def find_best_alignment(canvas, patch):
    """Generically aligns a new patch onto the canvas by finding the translation (dx, dy)

    and slight angular tweak that maximizes edge boundary contact while minimizing overlap.
    """
    m_canvas = get_mask(canvas)
    m_patch = get_mask(patch)

    # Edge contours of patch
    canny_p = cv2.Canny(m_patch, 50, 150)
    y_p, x_p = np.where(canny_p == 255)

    if len(y_p) == 0:
        return 0, 0, float("inf")

    # Downsample points for speed
    if len(y_p) > 350:
        idx = np.linspace(0, len(y_p) - 1, 350, dtype=int)
        y_p, x_p = y_p[idx], x_p[idx]

    dist_out = cv2.distanceTransform(~m_canvas, cv2.DIST_L2, 5)
    dist_in = cv2.distanceTransform(m_canvas, cv2.DIST_L2, 5)

    h_c, w_c = canvas.shape[:2]
    h_p, w_p = patch.shape[:2]

    best_dx, best_dy = 0, 0
    min_score = float("inf")

    # Coarse search across potential bounding overlaps
    step = 8
    for dy in range(-int(h_p * 0.7), int(h_c * 0.9), step):
        for dx in range(-int(w_p * 0.7), int(w_c * 0.9), step):
            sy, sx = y_p + dy, x_p + dx
            valid = (sy >= 0) & (sy < h_c) & (sx >= 0) & (sx < w_c)
            if np.sum(valid) < 10:
                continue

            vy, vx = sy[valid], sx[valid]
            overlap = m_canvas[vy, vx] == 255
            non_overlap_y, non_overlap_x = vy[~overlap], vx[~overlap]

            # Reward edges close to existing boundaries, heavily penalize deep overlaps
            touching = (
                np.sum(dist_out[non_overlap_y, non_overlap_x] <= 5.0)
                if len(non_overlap_y) > 0
                else 0
            )
            deep_overlap = (
                np.sum(dist_in[vy[overlap], vx[overlap]] > 6.0)
                if np.sum(overlap) > 0
                else 0
            )

            score = -touching + (5.0 * deep_overlap)

            if score < min_score:
                min_score = score
                best_dx, best_dy = dx, dy

    # Fine search refinement
    fine_step = 1
    for dy in range(best_dy - step, best_dy + step + 1, fine_step):
        for dx in range(best_dx - step, best_dx + step + 1, fine_step):
            sy, sx = y_p + dy, x_p + dx
            valid = (sy >= 0) & (sy < h_c) & (sx >= 0) & (sx < w_c)
            if np.sum(valid) < 10:
                continue

            vy, vx = sy[valid], sx[valid]
            overlap = m_canvas[vy, vx] == 255
            non_overlap_y, non_overlap_x = vy[~overlap], vx[~overlap]

            touching = (
                np.sum(dist_out[non_overlap_y, non_overlap_x] <= 5.0)
                if len(non_overlap_y) > 0
                else 0
            )
            deep_overlap = (
                np.sum(dist_in[vy[overlap], vx[overlap]] > 6.0)
                if np.sum(overlap) > 0
                else 0
            )

            score = -touching + (5.0 * deep_overlap)

            if score < min_score:
                min_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, min_score


def merge_patch_to_canvas(canvas, patch, dx, dy):
    """Generically expands canvas if needed and overlays patch onto canvas at (dx, dy)."""
    h_c, w_c = canvas.shape[:2]
    h_p, w_p = patch.shape[:2]

    m_patch = get_mask(patch)

    oy, ox = max(0, -dy), max(0, -dx)
    new_h = max(h_c + oy, h_p + dy + oy)
    new_w = max(w_c + ox, w_p + dx + ox)

    new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # Place existing canvas
    new_canvas[oy : oy + h_c, ox : ox + w_c] = canvas

    # Place new patch
    patch_target_y = dy + oy
    patch_target_x = dx + ox

    patch_area = new_canvas[
        patch_target_y : patch_target_y + h_p,
        patch_target_x : patch_target_x + w_p,
    ]
    patch_area[m_patch == 255] = patch[m_patch == 255]

    return new_canvas


def display_final_canvas(raw_fragment_metadata):
    """Fully generalized dynamic stitching loop using greedy best-match edge search."""
    if not raw_fragment_metadata:
        return None

    unstitched = [item["img"] for item in raw_fragment_metadata]

    # Sort fragments by size (area) to start with the largest piece as base canvas
    unstitched.sort(
        key=lambda img: np.count_nonzero(get_mask(img)), reverse=True
    )

    canvas = unstitched.pop(0)

    # Greedily pick and stitch the next best-fitting fragment
    while unstitched:
        best_candidate_idx = 0
        best_dx, best_dy = 0, 0
        best_score = float("inf")

        for idx, patch in enumerate(unstitched):
            dx, dy, score = find_best_alignment(canvas, patch)
            if score < best_score:
                best_score = score
                best_dx, best_dy = dx, dy
                best_candidate_idx = idx

        # Merge the winning piece
        winning_patch = unstitched.pop(best_candidate_idx)
        canvas = merge_patch_to_canvas(canvas, winning_patch, best_dx, best_dy)

    return canvas


def apply_inpainting_transformation(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    paper_mask = (gray > 15).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_paper = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel)
    gaps_mask = cv2.bitwise_and(closed_paper, cv2.bitwise_not(paper_mask))

    _, bright_seams = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    seams_mask = cv2.bitwise_or(gaps_mask, bright_seams)
    seams_mask = cv2.bitwise_and(seams_mask, closed_paper)

    inpainted = cv2.inpaint(
        canvas, seams_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )

    lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(
        enhanced_gray,
        200,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        3,
        2,
    )

    return binarized


class ApplicationWindow:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Document Fragment Stitcher")
        self.root.geometry("560x540")
        self.root.configure(bg="#0F172A")
        self.root.resizable(False, False)

        self.selected_path = None
        self.input_preview_photo = None
        self.setup_ui()

    def setup_ui(self):
        header = tk.Frame(self.root, bg="#1E293B", height=60)
        header.pack(fill=tk.X, side=tk.TOP)

        lbl_title = tk.Label(
            header,
            text="✨ Document Reconstruction Engine",
            font=("Segoe UI", 15, "bold"),
            bg="#1E293B",
            fg="#38BDF8",
        )
        lbl_title.pack(pady=12)

        body = tk.Frame(self.root, bg="#0F172A", padx=25, pady=20)
        body.pack(fill=tk.BOTH, expand=True)

        file_frame = tk.Frame(
            body, bg="#1E293B", bd=1, relief=tk.SOLID, padx=12, pady=10
        )
        file_frame.pack(fill=tk.X, pady=(0, 15))

        self.btn_browse = tk.Button(
            file_frame,
            text="📁 Browse Original Image",
            font=("Segoe UI", 10, "bold"),
            bg="#2563EB",
            fg="white",
            activebackground="#1D4ED8",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.browse_file,
        )
        self.btn_browse.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_path = tk.Label(
            file_frame,
            text="No image selected...",
            font=("Segoe UI", 9, "italic"),
            bg="#1E293B",
            fg="#94A3B8",
            anchor="w",
        )
        self.lbl_path.pack(side=tk.LEFT, fill=tk.X, expand=True)

        preview_card = tk.LabelFrame(
            body,
            text=" Original Image Preview ",
            font=("Segoe UI", 10, "bold"),
            bg="#1E293B",
            fg="#F1F5F9",
            bd=1,
            relief=tk.SOLID,
            padx=10,
            pady=10,
        )
        preview_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        self.lbl_preview = tk.Label(
            preview_card,
            text=(
                "🖼️ No image loaded yet\nClick 'Browse Original Image' above"
                " to load"
            ),
            font=("Segoe UI", 10),
            bg="#0F172A",
            fg="#64748B",
        )
        self.lbl_preview.pack(fill=tk.BOTH, expand=True)

        self.btn_convert = tk.Button(
            body,
            text="🧩 Stitch Fragments",
            font=("Segoe UI", 11, "bold"),
            bg="#10B981",
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            disabledforeground="#475569",
            bd=0,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.process_image,
        )
        self.btn_convert.pack(fill=tk.X, pady=(0, 10))

        self.lbl_status = tk.Label(
            body,
            text="Status: Ready",
            font=("Segoe UI", 9, "bold"),
            bg="#0F172A",
            fg="#94A3B8",
        )
        self.lbl_status.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File from PC",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.selected_path = file_path
            self.lbl_path.config(
                text=os.path.basename(file_path), fg="#F8FAFC"
            )
            self.btn_convert.config(state=tk.NORMAL, bg="#10B981")
            self.lbl_status.config(
                text="Status: Original Image Loaded. Click 'Stitch Fragments'!",
                fg="#38BDF8",
            )

            pil_img = Image.open(file_path)
            pil_img.thumbnail((260, 190))
            self.input_preview_photo = ImageTk.PhotoImage(pil_img)
            self.lbl_preview.config(
                image=self.input_preview_photo, text="", bg="#1E293B"
            )

    def process_image(self):
        if not self.selected_path:
            return

        self.btn_convert.config(state=tk.DISABLED, bg="#334155")
        self.btn_browse.config(state=tk.DISABLED)
        self.lbl_status.config(
            text="Status: ⏳ Dynamically matching & stitching fragments...",
            fg="#FBBF24",
        )
        self.root.update()

        raw_fragment_metadata, _ = display_detected_fragments(
            self.selected_path
        )
        if not raw_fragment_metadata:
            messagebox.showerror(
                "Error",
                "No distinct paper fragments found or image path error.",
            )
            self.reset_ui()
            return

        stitched_canvas = display_final_canvas(raw_fragment_metadata)

        self.root.withdraw()
        ResultViewer(self.root, stitched_canvas)

    def reset_ui(self):
        self.btn_convert.config(state=tk.NORMAL, bg="#10B981")
        self.btn_browse.config(state=tk.NORMAL)
        self.lbl_status.config(text="Status: Ready", fg="#94A3B8")

    def run(self):
        self.root.mainloop()


class ResultViewer:

    def __init__(self, main_root, stitched_img):
        self.main_root = main_root
        self.stitched_img = stitched_img
        self.current_display_img = stitched_img
        self.inpainted_img = None

        self.window = tk.Toplevel()
        self.window.title("Stitched Canvas Viewer")
        self.window.geometry("820x700")
        self.window.configure(bg="#0F172A")
        self.window.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.setup_ui()

    def setup_ui(self):
        header = tk.Frame(self.window, bg="#1E293B", padx=15, pady=10)
        header.pack(fill=tk.X, side=tk.TOP)

        self.btn_inpaint = tk.Button(
            header,
            text="✨ Apply Inpainting",
            font=("Segoe UI", 10, "bold"),
            bg="#8B5CF6",
            fg="white",
            activebackground="#7C3AED",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.trigger_inpainting,
        )
        self.btn_inpaint.pack(side=tk.LEFT, padx=(0, 10))

        btn_save = tk.Button(
            header,
            text="💾 Save Image",
            font=("Segoe UI", 10, "bold"),
            bg="#059669",
            fg="white",
            activebackground="#047857",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.save_image,
        )
        btn_save.pack(side=tk.LEFT)

        self.title_lbl = tk.Label(
            header,
            text="Stitched Image (Generalized Dynamic Stitcher)",
            font=("Segoe UI", 12, "bold"),
            bg="#1E293B",
            fg="#38BDF8",
        )
        self.title_lbl.pack(side=tk.LEFT, expand=True)

        btn_exit = tk.Button(
            header,
            text="❌ Exit",
            font=("Segoe UI", 10, "bold"),
            bg="#E11D48",
            fg="white",
            activebackground="#BE123C",
            activeforeground="white",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.exit_app,
        )
        btn_exit.pack(side=tk.RIGHT)

        self.img_card = tk.Frame(
            self.window, bg="#1E293B", bd=1, relief=tk.SOLID
        )
        self.img_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.lbl_image = tk.Label(self.img_card, bg="#0F172A", bd=0)
        self.lbl_image.pack(expand=True, anchor="center", padx=10, pady=10)

        self.render_image(self.stitched_img)

    def render_image(self, img_array):
        if len(img_array.shape) == 2:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((760, 550))
        self.photo = ImageTk.PhotoImage(pil_img)

        self.lbl_image.config(image=self.photo)

    def trigger_inpainting(self):
        if self.inpainted_img is None:
            self.title_lbl.config(
                text="Processing Inpainting...", fg="#FBBF24"
            )
            self.window.update()

            self.inpainted_img = apply_inpainting_transformation(
                self.stitched_img
            )

        self.current_display_img = self.inpainted_img
        self.render_image(self.inpainted_img)
        self.title_lbl.config(
            text="Stitched Image with Inpainting Applied", fg="#10B981"
        )

        self.btn_inpaint.config(
            state=tk.DISABLED, bg="#475569", text="✓ Inpainting Applied"
        )

    def save_image(self):
        if self.current_display_img is None:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("All Files", "*.*"),
            ],
            title="Save Output Image",
        )
        if file_path:
            cv2.imwrite(file_path, self.current_display_img)
            messagebox.showinfo(
                "Saved", f"File successfully saved to:\n{file_path}"
            )

    def exit_app(self):
        self.window.destroy()
        self.main_root.destroy()
        sys.exit()


if __name__ == "__main__":
    app = ApplicationWindow()
    app.run()