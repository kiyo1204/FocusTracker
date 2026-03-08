import tkinter as tk
from tkinter import scrolledtext
import subprocess
import threading
import os

class ScriptControllerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("集中度計測 AIコントローラー")
        self.geometry("+250-100")

        # 現在実行中のプロセスを保存する変数
        self.current_process = None 

        # --- UI部品の配置 ---
        self.left_target_script = "./Scripts/data_collection.py"
        self.right_target_script = "./Scripts/train.py"

        # 1. ボタン全体をまとめる大枠のフレーム
        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)

        # 2. 左列用のフレーム
        left_col = tk.Frame(button_frame)
        left_col.pack(side=tk.LEFT, padx=10, anchor=tk.N)

        # 3. 右列用のフレーム
        right_col = tk.Frame(button_frame)
        right_col.pack(side=tk.LEFT, padx=10, anchor=tk.N)

        # ラベル
        self.label_left = tk.Label(left_col, text="【データ収集・計測】", font=("MS Gothic", 10, "bold"))
        self.label_left.pack(pady=(0, 10))

        # --- 左列のボタン ---
        self.btn_mode1 = tk.Button(
            left_col,
            text="① 集中データを収集", 
            command=lambda: self.run_script(self.left_target_script, "1")
        )
        self.btn_mode1.pack(pady=5, fill=tk.X)

        self.btn_mode2 = tk.Button(
            left_col,
            text="② 非集中データを収集",
            command=lambda: self.run_script(self.left_target_script, "2")
        )
        self.btn_mode2.pack(pady=5, fill=tk.X)

        self.btn_mode3 = tk.Button(
            left_col,
            text="③ 計測モード (予測)",
            command=lambda: self.run_script(self.left_target_script, "3"), bg="lightgreen"
        )
        self.btn_mode3.pack(pady=5, fill=tk.X)

        self.label_right = tk.Label(right_col, text="【モデルの作成・保存】", font=("MS Gothic", 10, "bold"))
        self.label_right.pack(pady=(0, 10))
        # --- 右列の追加ボタン ---
        self.btn_mode4 = tk.Button(
            right_col,
            text="モデル作成・保存",
            command=lambda: self.run_script(self.right_target_script, "train")
        )
        self.btn_mode4.pack(pady=5, fill=tk.X)

        # 実行結果を表示するテキストエリア
        self.log_area = scrolledtext.ScrolledText(self, width=100, height=30, state='disabled')
        self.log_area.pack(pady=10)

        # スクリプトへ送信するための入力欄とボタン
        input_frame = tk.Frame(self)
        input_frame.pack(pady=5)
        
        self.entry_input = tk.Entry(input_frame, width=30)
        self.entry_input.pack(side=tk.LEFT, padx=5)
        
        self.btn_send = tk.Button(input_frame, text="スクリプトへ送信", command=self.send_input)
        self.btn_send.pack(side=tk.LEFT)

    def write_log(self, message):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    # 入力された文字をプロセスに送る関数
    def send_input(self):
        if self.current_process and self.current_process.poll() is None: # プロセスが実行中か確認
            text_to_send = self.entry_input.get()
            try:
                # プロセスの標準入力 (stdin) に文字を書き込んで、Enterキー(\n)を押すイメージ
                self.current_process.stdin.write(text_to_send + "\n")
                self.current_process.stdin.flush() # 溜め込まずに即座に送信
                self.write_log(f"> 送信: {text_to_send}")
                self.entry_input.delete(0, tk.END) # 入力欄を空にする
            except Exception as e:
                self.write_log(f"送信エラー: {e}")
        else:
            self.write_log("エラー: スクリプトが実行されていません。")

    def run_script(self, script_name, arg=None):
        mode_text = f" (モード: {arg})" if arg else ""
        self.write_log(f"\n--- [{script_name[10:]}]{mode_text} の実行を開始 ---")
        
        def task():
            try:
                command = ["python", "-u", script_name]
                if arg:
                    command.append(arg)

                my_env = os.environ.copy()
                my_env["PYTHONIOENCODING"] = "utf-8"

                self.current_process = subprocess.Popen(
                    command, 
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True, 
                    encoding='utf-8',
                    errors='replace',
                    env=my_env
                )
                
                for line in self.current_process.stdout:
                    clean_line = line.strip()
                    if not clean_line: continue
                    if "INFO:" in clean_line or "WARNING:" in clean_line or "Warning" in clean_line or "To enable" in clean_line: continue
                    if clean_line.startswith("I0") or clean_line.startswith("W0"): continue
                    self.write_log(clean_line)
                
                self.current_process.wait()
                self.write_log("--- 完了 ---")
            
            except Exception as e:
                self.write_log(f"予期せぬエラー: {e}")

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

if __name__ == "__main__":
    app = ScriptControllerApp()
    app.mainloop()