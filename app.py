import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import subprocess
import threading
import os

class ScriptControllerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Focus Tracker")

        # 使っているPCの画面の幅と高さを取得して最大化
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}+0+0")

        self.current_process = None 
        self.left_target_script = "./Scripts/data_collection.py"
        self.right_target_script = "./Scripts/train.py"

        # アプリ全体を閉じたときの安全処理
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ==========================================
        # 大枠の画面（土台）を作成
        # ==========================================
        self.frame_top = tk.Frame(self)
        self.frame_pred = tk.Frame(self)
        self.frame_dev = tk.Frame(self)

        # 各画面を構築
        self.build_top_screen()
        self.build_pred_screen()
        self.build_dev_screen()

        # 最初はトップ画面を表示
        self.show_screen(self.frame_top)

    # ==========================================
    # プロセス管理と画面遷移の安全処理
    # ==========================================
    def stop_current_process(self):
        # 現在実行中のプロセスがあれば強制終了する
        if self.current_process and self.current_process.poll() is None:
            try:
                self.current_process.terminate()
            except Exception as e:
                print(f"プロセスの終了に失敗しました: {e}")
            finally:
                self.current_process = None

    def back_to_top(self):
        # トップ画面に戻る前に、動いているプロセスを安全に止める
        self.stop_current_process()
        self.show_screen(self.frame_top)

    def on_tab_changed(self, event):
        # タブが切り替わった時にプロセスをリセットする
        self.stop_current_process()

    def on_closing(self):
        self.stop_current_process()
        self.destroy()

    def show_screen(self, frame):
        # 指定された画面のみを表示
        self.frame_top.pack_forget()
        self.frame_pred.pack_forget()
        self.frame_dev.pack_forget()
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # ==========================================
    # トップ画面の構築
    # ==========================================
    def build_top_screen(self):
        lbl_title = tk.Label(self.frame_top, text="Focus Tracker", font=("MS Gothic", 32, "bold"))
        lbl_title.pack(pady=80)

        btn_pred = tk.Button(
            self.frame_top, text="予測モード\n(計測のみを行うシンプルな画面)", 
            font=("MS Gothic", 16), bg="lightgreen", width=35, height=3,
            command=lambda: self.show_screen(self.frame_pred)
        )
        btn_pred.pack(pady=15)

        btn_dev = tk.Button(
            self.frame_top, text="開発者モード\n(データ収集・学習・ログ監視)", 
            font=("MS Gothic", 16), width=35, height=3,
            command=lambda: self.show_screen(self.frame_dev)
        )
        btn_dev.pack(pady=15)

    # ==========================================
    # 予測モード画面の構築
    # ==========================================
    def build_pred_screen(self):
        btn_back = tk.Button(self.frame_pred, text="◀ トップに戻る", font=("MS Gothic", 12), command=self.back_to_top)
        btn_back.pack(anchor=tk.NW, pady=(0, 10))

        lbl_title = tk.Label(self.frame_pred, text="【 予測モード 】", font=("MS Gothic", 20, "bold"))
        lbl_title.pack(pady=10)

        btn_start = tk.Button(
            self.frame_pred, text="計測スタート", font=("MS Gothic", 16, "bold"), 
            bg="lightgreen", width=25, height=2,
            command=lambda: self.run_script(self.left_target_script, "3", self.log_area_pred)
        )
        btn_start.pack(pady=15)

        self.log_area_pred = scrolledtext.ScrolledText(self.frame_pred, width=80, height=20, state='disabled', font=("MS Gothic", 14))
        self.log_area_pred.pack(pady=10)

    # ==========================================
    # 開発者モード画面の構築 (ここにタブを組み込む)
    # ==========================================
    def build_dev_screen(self):
        btn_back = tk.Button(self.frame_dev, text="◀ トップに戻る", font=("MS Gothic", 12), command=self.back_to_top)
        btn_back.pack(anchor=tk.NW, pady=(0, 10))

        # タブのスタイル設定
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook.Tab', font=('MS Gothic', 14, 'bold'), padding=[20, 10])

        self.notebook = ttk.Notebook(self.frame_dev)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # タブ切り替え時の安全処理をバインド
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # タブ1とタブ2の土台を作成
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_data, text=" ① 学習データ取得・計測 ")
        self.notebook.add(self.tab_train, text=" ② モデル学習 ")

        # 各タブの中身を構築
        self.build_data_tab()
        self.build_train_tab()

    # --- データ収集ページ ---
    def build_data_tab(self):
        lbl_title = tk.Label(self.tab_data, text="【 学習データの収集とリアルタイム計測 】", font=("MS Gothic", 16, "bold"))
        lbl_title.pack(pady=15)

        btn_frame = tk.Frame(self.tab_data)
        btn_frame.pack(pady=10)

        btn_mode1 = tk.Button(
            btn_frame, text="① 集中データ", font=("MS Gothic", 12), width=20, height=2,
            command=lambda: self.run_script(self.left_target_script, "1", self.log_area_data)
        )
        btn_mode1.pack(side=tk.LEFT, padx=10)

        btn_mode2 = tk.Button(
            btn_frame, text="② 非集中データ", font=("MS Gothic", 12), width=20, height=2,
            command=lambda: self.run_script(self.left_target_script, "2", self.log_area_data)
        )
        btn_mode2.pack(side=tk.LEFT, padx=10)


        self.log_area_data = scrolledtext.ScrolledText(self.tab_data, width=100, height=20, state='disabled', font=("MS Gothic", 14))
        self.log_area_data.pack(pady=15, fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(self.tab_data)
        input_frame.pack(pady=10)
        
        self.entry_input_data = tk.Entry(input_frame, font=("MS Gothic", 14), width=40)
        self.entry_input_data.pack(side=tk.LEFT, padx=5)
        
        btn_send = tk.Button(input_frame, text="スクリプトへ送信", font=("MS Gothic", 12), 
                            command=lambda: self.send_input(self.entry_input_data, self.log_area_data))
        btn_send.pack(side=tk.LEFT)

    # --- モデル作成ページ ---
    def build_train_tab(self):
        lbl_title = tk.Label(self.tab_train, text="【 AIモデルの学習と保存 】", font=("MS Gothic", 16, "bold"))
        lbl_title.pack(pady=15)

        btn_train = tk.Button(
            self.tab_train, text="モデル学習を開始", font=("MS Gothic", 14, "bold"), bg="lightblue", width=25, height=2,
            command=lambda: self.run_script(self.right_target_script, "train", self.log_area_train)
        )
        btn_train.pack(pady=10)

        self.log_area_train = scrolledtext.ScrolledText(self.tab_train, width=100, height=20, state='disabled', font=("MS Gothic", 14))
        self.log_area_train.pack(pady=15, fill=tk.BOTH, expand=True)

    # 共通処理: ログ出力・プロセス実行
    def write_log(self, message, target_log_area):
        target_log_area.config(state='normal')
        target_log_area.insert(tk.END, message + "\n")
        target_log_area.see(tk.END)
        target_log_area.config(state='disabled')

    def send_input(self, entry_widget, target_log_area):
        if self.current_process and self.current_process.poll() is None:
            text_to_send = entry_widget.get()
            try:
                self.current_process.stdin.write(text_to_send + "\n")
                self.current_process.stdin.flush()
                self.write_log(f"> 送信: {text_to_send}", target_log_area)
                entry_widget.delete(0, tk.END)
            except Exception as e:
                self.write_log(f"送信エラー: {e}", target_log_area)
        else:
            self.write_log("エラー: スクリプトが実行されていません。", target_log_area)

    def run_script(self, script_name, arg=None, target_log_area=None):
        self.stop_current_process()

        if arg == "1":
            execute_script = "集中データ収集"
        elif arg == "2":
            execute_script = "非集中データ収集"
        elif arg == "3":
            execute_script = "計測"
        else:
            execute_script = "モデルの学習"

        self.write_log(f"--- [{execute_script}]を実行 ---", target_log_area)
        self.write_log("セットアップ中です. しばらくお待ちください.", target_log_area)
        
        def task():
            try:
                command = ["python", "-u", script_name]
                if arg:
                    command.append(arg)

                my_env = os.environ.copy()
                my_env["PYTHONIOENCODING"] = "utf-8"

                process = subprocess.Popen(
                    command, 
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True, 
                    encoding='utf-8',
                    errors='replace',
                    env=my_env
                )
                
                self.current_process = process
                
                for line in process.stdout:
                    clean_line = line.strip()
                    if not clean_line: continue
                    if "INFO:" in clean_line or "WARNING:" in clean_line or "Warning" in clean_line or "To enable" in clean_line: continue
                    if clean_line.startswith("I0") or clean_line.startswith("W0"): continue
                    self.write_log(clean_line, target_log_area)
                
                process.wait()
                self.write_log("--- 終了 ---", target_log_area)

            except Exception as e:
                self.write_log(f"予期せぬエラー: {e}", target_log_area)

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

if __name__ == "__main__":
    app = ScriptControllerApp()
    app.mainloop()