"""
app.py — 크로마키 배경 제거기 GUI
=====================================
드래그앤드롭 또는 버튼으로 영상 추가 → 초록 배경 .mp4 일괄 변환
"""

import sys
import os
import threading
import subprocess
import shutil
import queue
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

# tkinterdnd2 임포트 (없으면 일반 tkinter 폴백)
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
MODEL_NAME   = "u2net"
GREEN_BG     = (0, 255, 0)
SUPPORTED    = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".m4v", ".webm"}
APP_VERSION  = "1.0.0"


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def get_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_providers():
    try:
        import onnxruntime as ort
        av = ort.get_available_providers()
        if "CUDAExecutionProvider" in av:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA GPU"
        if "DmlExecutionProvider" in av:
            return ["DmlExecutionProvider", "CPUExecutionProvider"], "DirectML GPU"
    except Exception:
        pass
    return ["CPUExecutionProvider"], "CPU"


def composite_green(rgba_arr: np.ndarray) -> np.ndarray:
    rgb   = rgba_arr[:, :, :3].astype(np.float32)
    alpha = rgba_arr[:, :, 3:4].astype(np.float32) / 255.0
    green = np.array(GREEN_BG, dtype=np.float32)
    blended = rgb * alpha + green * (1.0 - alpha)
    return blended[:, :, ::-1].astype(np.uint8)


# ─────────────────────────────────────────────
# 변환 워커 (별도 스레드)
# ─────────────────────────────────────────────
def convert_worker(files, output_dir, ffmpeg_bin, session, msg_q):
    """
    msg_q 에 넣는 메시지 형식:
      ("file_start",  filename, total_frames)
      ("frame",       filename, current_frame)
      ("file_done",   filename, out_path)
      ("file_error",  filename, error_msg)
      ("all_done",    success_count, fail_count)
    """
    success, failed = 0, 0
    for video_path in files:
        video_path = Path(video_path)
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("영상을 열 수 없습니다.")

            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = Path(output_dir) / f"green_{video_path.stem}.mp4"
            msg_q.put(("file_start", video_path.name, total))

            ffmpeg_cmd = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-preset", "fast",
                "-an",
                str(out_path),
            ]
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            frame_idx = 0
            try:
                while True:
                    ret, bgr = cap.read()
                    if not ret:
                        break
                    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgba = remove(Image.fromarray(rgb), session=session)
                    proc.stdin.write(composite_green(np.array(rgba, dtype=np.uint8)).tobytes())
                    frame_idx += 1
                    msg_q.put(("frame", video_path.name, frame_idx))
            finally:
                cap.release()
                proc.stdin.close()
                _, stderr = proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(stderr.decode(errors="replace")[:200])

            msg_q.put(("file_done", video_path.name, str(out_path)))
            success += 1

        except Exception as e:
            msg_q.put(("file_error", video_path.name, str(e)))
            failed += 1

    msg_q.put(("all_done", success, failed))


# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────
class App:
    # 색상 팔레트
    BG        = "#1a1a2e"
    PANEL     = "#16213e"
    ACCENT    = "#00d4aa"
    ACCENT2   = "#0f3460"
    TEXT      = "#e0e0e0"
    TEXT_DIM  = "#888888"
    SUCCESS   = "#00c853"
    ERROR     = "#ff5252"
    WARNING   = "#ffab40"
    DROP_BG   = "#0d1b2a"

    def __init__(self, root):
        self.root = root
        self.root.title(f"크로마키 배경 제거기  v{APP_VERSION}")
        self.root.geometry("780x620")
        self.root.minsize(680, 520)
        self.root.configure(bg=self.BG)

        # 상태
        self.files        = []          # 대기 파일 목록
        self.converting   = False
        self.msg_q        = queue.Queue()
        self.session      = None
        self.ffmpeg_bin   = None
        self.output_dir   = Path.home() / "Desktop" / "green_output"
        self.total_frames = {}          # filename → total
        self.cur_frames   = {}          # filename → done

        self._build_ui()
        self._init_backend()           # 백그라운드에서 모델 로드
        self.root.after(100, self._poll_queue)

    # ── UI 구성 ──────────────────────────────
    def _build_ui(self):
        # 헤더
        hdr = tk.Frame(self.root, bg=self.ACCENT2, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎬  크로마키 배경 제거기",
                 font=("Segoe UI", 16, "bold"),
                 bg=self.ACCENT2, fg=self.ACCENT).pack(side="left", padx=20)
        self.lbl_gpu = tk.Label(hdr, text="모델 로딩 중...",
                                font=("Segoe UI", 9),
                                bg=self.ACCENT2, fg=self.TEXT_DIM)
        self.lbl_gpu.pack(side="right", padx=16)

        # 드롭 존
        drop_frame = tk.Frame(self.root, bg=self.BG, pady=10)
        drop_frame.pack(fill="x", padx=20)

        self.drop_zone = tk.Label(
            drop_frame,
            text="여기에 영상 파일을 드래그하거나\n아래 버튼으로 파일을 선택하세요\n\n지원 형식: mp4  mov  avi  mkv  webm",
            font=("Segoe UI", 11),
            bg=self.DROP_BG, fg=self.TEXT_DIM,
            relief="flat", bd=0,
            pady=28,
        )
        self.drop_zone.pack(fill="x")

        # 드래그앤드롭 등록
        if HAS_DND:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<Drop>>", self._on_drop)
            self.drop_zone.configure(text=(
                "📂  영상 파일을 여기에 드래그하세요\n\n"
                "또는 아래 버튼으로 파일 선택\n\n"
                "지원 형식: mp4  mov  avi  mkv  webm"
            ))

        # 버튼 행
        btn_row = tk.Frame(self.root, bg=self.BG)
        btn_row.pack(fill="x", padx=20, pady=(6, 0))

        self._btn(btn_row, "📁  파일 선택", self._pick_files).pack(side="left", padx=(0, 8))
        self._btn(btn_row, "📂  출력 폴더 변경", self._pick_output).pack(side="left", padx=(0, 8))
        self._btn(btn_row, "🗑  목록 초기화", self._clear_list, color="#444").pack(side="left")

        self.lbl_outdir = tk.Label(btn_row,
                                   text=f"출력: {self.output_dir}",
                                   font=("Segoe UI", 8), bg=self.BG, fg=self.TEXT_DIM)
        self.lbl_outdir.pack(side="right")

        # 파일 목록
        list_frame = tk.Frame(self.root, bg=self.BG)
        list_frame.pack(fill="both", expand=True, padx=20, pady=(10, 0))

        self.listbox = tk.Listbox(
            list_frame,
            bg=self.PANEL, fg=self.TEXT,
            selectbackground=self.ACCENT2,
            font=("Segoe UI", 10),
            relief="flat", bd=0,
            activestyle="none",
        )
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.listbox.pack(fill="both", expand=True)

        # 현재 파일 진행바
        prog_frame = tk.Frame(self.root, bg=self.BG, pady=6)
        prog_frame.pack(fill="x", padx=20)

        self.lbl_cur = tk.Label(prog_frame, text="대기 중",
                                font=("Segoe UI", 9), bg=self.BG, fg=self.TEXT_DIM)
        self.lbl_cur.pack(anchor="w")

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Green.Horizontal.TProgressbar",
                        troughcolor=self.PANEL,
                        background=self.ACCENT,
                        thickness=12)
        self.prog_file = ttk.Progressbar(prog_frame, style="Green.Horizontal.TProgressbar",
                                         orient="horizontal", mode="determinate")
        self.prog_file.pack(fill="x", pady=(2, 0))

        # 전체 진행바
        self.lbl_total = tk.Label(prog_frame, text="",
                                  font=("Segoe UI", 9), bg=self.BG, fg=self.TEXT_DIM)
        self.lbl_total.pack(anchor="w", pady=(4, 0))

        style.configure("Total.Horizontal.TProgressbar",
                        troughcolor=self.PANEL,
                        background=self.ACCENT2,
                        thickness=8)
        self.prog_total = ttk.Progressbar(prog_frame, style="Total.Horizontal.TProgressbar",
                                          orient="horizontal", mode="determinate")
        self.prog_total.pack(fill="x", pady=(2, 0))

        # 하단 버튼
        bot = tk.Frame(self.root, bg=self.BG, pady=12)
        bot.pack(fill="x", padx=20)

        self.btn_start = self._btn(bot, "▶  변환 시작", self._start_convert,
                                   color=self.ACCENT, fg="#000", width=18,
                                   font=("Segoe UI", 11, "bold"))
        self.btn_start.pack(side="left")
        self.btn_start.configure(state="disabled")

        self._btn(bot, "📂  결과 폴더 열기", self._open_output).pack(side="right")

    def _btn(self, parent, text, cmd, color=None, fg=None, width=None, font=None):
        c  = color or self.ACCENT2
        f  = fg    or self.TEXT
        fn = font  or ("Segoe UI", 10)
        kw = dict(text=text, command=cmd, bg=c, fg=f,
                  font=fn, relief="flat", bd=0,
                  padx=14, pady=7, cursor="hand2",
                  activebackground=self.ACCENT, activeforeground="#000")
        if width:
            kw["width"] = width
        return tk.Button(parent, **kw)

    # ── 백엔드 초기화 (스레드) ───────────────
    def _init_backend(self):
        def _load():
            self.ffmpeg_bin = get_ffmpeg()
            providers, gpu_name = get_providers()
            self.session = new_session(MODEL_NAME, providers=providers)
            self.msg_q.put(("ready", gpu_name))
        threading.Thread(target=_load, daemon=True).start()

    # ── 드롭/파일 선택 ───────────────────────
    def _on_drop(self, event):
        raw = event.data
        # tkinterdnd2 는 중괄호로 묶인 경로 반환 가능
        import re
        paths = re.findall(r'\{([^}]+)\}|(\S+)', raw)
        for a, b in paths:
            p = a or b
            if Path(p).suffix.lower() in SUPPORTED:
                self._add_file(p)

    def _pick_files(self):
        paths = filedialog.askopenfilenames(
            title="영상 파일 선택",
            filetypes=[("영상 파일", "*.mp4 *.mov *.avi *.mkv *.wmv *.webm *.flv"), ("모든 파일", "*.*")]
        )
        for p in paths:
            self._add_file(p)

    def _pick_output(self):
        d = filedialog.askdirectory(title="출력 폴더 선택")
        if d:
            self.output_dir = Path(d)
            self.lbl_outdir.configure(text=f"출력: {self.output_dir}")

    def _add_file(self, path):
        if path not in self.files:
            self.files.append(path)
            name = Path(path).name
            self.listbox.insert("end", f"  ⏳  {name}")
        if self.session and not self.converting:
            self.btn_start.configure(state="normal")

    def _clear_list(self):
        if self.converting:
            return
        self.files.clear()
        self.listbox.delete(0, "end")
        self.btn_start.configure(state="disabled")
        self.prog_file["value"]  = 0
        self.prog_total["value"] = 0
        self.lbl_cur.configure(text="대기 중")
        self.lbl_total.configure(text="")

    # ── 변환 시작 ────────────────────────────
    def _start_convert(self):
        if not self.files or self.converting:
            return
        if not self.ffmpeg_bin:
            messagebox.showerror("오류", "ffmpeg 를 찾을 수 없습니다.\npip install imageio-ffmpeg 를 실행하세요.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converting = True
        self.btn_start.configure(state="disabled", text="변환 중...")
        self.total_frames = {}
        self.cur_frames   = {}
        self.prog_total["maximum"] = len(self.files)
        self.prog_total["value"]   = 0

        t = threading.Thread(
            target=convert_worker,
            args=(list(self.files), self.output_dir,
                  self.ffmpeg_bin, self.session, self.msg_q),
            daemon=True,
        )
        t.start()

    # ── 큐 폴링 (100ms마다) ──────────────────
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _handle_msg(self, msg):
        kind = msg[0]

        if kind == "ready":
            gpu = msg[1]
            self.lbl_gpu.configure(text=f"✅  {gpu}  |  모델: {MODEL_NAME}", fg=self.ACCENT)
            if self.files:
                self.btn_start.configure(state="normal")

        elif kind == "file_start":
            _, name, total = msg
            self.total_frames[name] = total
            self.cur_frames[name]   = 0
            self.prog_file["maximum"] = total
            self.prog_file["value"]   = 0
            self.lbl_cur.configure(text=f"처리 중: {name}  (0 / {total} 프레임)", fg=self.TEXT)
            # 목록에서 해당 항목 색 변경
            self._set_list_item(name, f"  🔄  {name}", self.WARNING)

        elif kind == "frame":
            _, name, cur = msg
            self.cur_frames[name] = cur
            total = self.total_frames.get(name, 1)
            self.prog_file["value"] = cur
            pct = int(cur / total * 100)
            self.lbl_cur.configure(text=f"처리 중: {name}  ({cur} / {total} 프레임  {pct}%)")

        elif kind == "file_done":
            _, name, out_path = msg
            self._set_list_item(name, f"  ✅  {name}", self.SUCCESS)
            done = sum(1 for v in self.cur_frames.values()
                       if v == self.total_frames.get(Path(v).name if isinstance(v, str) else "", -1))
            self.prog_total["value"] += 1
            self.prog_file["value"]   = self.prog_file["maximum"]

        elif kind == "file_error":
            _, name, err = msg
            self._set_list_item(name, f"  ❌  {name}  ({err[:40]})", self.ERROR)
            self.prog_total["value"] += 1

        elif kind == "all_done":
            _, success, failed = msg
            self.converting = False
            self.btn_start.configure(state="normal", text="▶  변환 시작")
            self.lbl_cur.configure(
                text=f"완료!  성공 {success}개  /  실패 {failed}개",
                fg=self.SUCCESS if failed == 0 else self.WARNING
            )
            if failed == 0:
                messagebox.showinfo("완료",
                    f"✅  {success}개 파일 변환 완료!\n\n출력 폴더:\n{self.output_dir}")
            else:
                messagebox.showwarning("완료 (일부 실패)",
                    f"성공: {success}개\n실패: {failed}개\n\n출력 폴더:\n{self.output_dir}")

    def _set_list_item(self, name, new_text, color):
        """listbox에서 파일명이 포함된 항목을 찾아 텍스트/색 갱신."""
        for i in range(self.listbox.size()):
            item = self.listbox.get(i)
            if name in item:
                self.listbox.delete(i)
                self.listbox.insert(i, new_text)
                self.listbox.itemconfigure(i, fg=color)
                break

    def _open_output(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.output_dir))


# ─────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────
def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    # 창 아이콘 (없어도 무방)
    try:
        root.iconbitmap(Path(__file__).parent / "icon.ico")
    except Exception:
        pass

    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
