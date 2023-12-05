import tkinter as tk
import time
import threading
def long_running_function():
    # Hàm mô phỏng một công việc mất thời gian
    for i in range(10):
        print(f"Đang thực hiện công việc {i}")
        time.sleep(10)
        root.update()  # Cập nhật cửa sổ Tkinter để hiển thị những thay đổi

def other_task():
    print("Thực hiện công việc khác")

def start_long_running_function():
    long_running_function()
    other_task()  # Hàm này sẽ được gọi khi long_running_function hoàn thành

root = tk.Tk()
root.minsize(width=1280,height=720)
button = tk.Button(root, text="Bắt đầu công việc dài hạn", command=start_long_running_function)
button.pack()

root.mainloop()
