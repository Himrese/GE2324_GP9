import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import math
import numpy as np
import joblib  # 用于加载 scaler

# -------------------------------
# 1. 创建主窗口（必须在创建 PhotoImage 前创建 root）
# -------------------------------
root = tk.Tk()
root.title("地图点击预测")
target_width = 800
target_height = 600
# 固定窗口大小，并预留按钮区域
root.geometry(f"{target_width}x{target_height+50}")

# -------------------------------
# 2. 加载训练好的模型（新模型输出为2个概率）
# -------------------------------
model = tf.keras.models.load_model("best_model.h5")

# -------------------------------
# 3. 加载归一化 scaler（训练时保存的 StandardScaler）
# -------------------------------
# 需在训练后保存 scaler，如：joblib.dump(scaler, 'scaler.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------------------
# 4. 加载并缩放地图图片
# -------------------------------
# 加载原始地图图片（请确保 "Abyss.png" 路径正确）
map_img_original = Image.open("Abyss.png")
original_width, original_height = map_img_original.size

# 计算原始坐标到缩放后坐标的比例（用于点击坐标转换）
scale_x = original_width / target_width
scale_y = original_height / target_height

# 使用 Image.LANCZOS 进行高质量缩放
map_img = map_img_original.resize((target_width, target_height), Image.LANCZOS)
tk_img = ImageTk.PhotoImage(map_img)

# -------------------------------
# 5. 创建 Canvas 显示地图
# -------------------------------
canvas = tk.Canvas(root, width=target_width, height=target_height)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=tk_img)

# 全局变量存储点击的原始坐标
clicked_x, clicked_y = None, None

# -------------------------------
# 6. 定义点击事件，转换点击坐标为原始坐标
# -------------------------------
def on_click(event):
    global clicked_x, clicked_y
    # event.x 和 event.y 是缩放后图片上的坐标，转换为原始坐标
    clicked_x = event.x * scale_x
    clicked_y = event.y * scale_y
    r = 5  # 标记半径
    canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="red")
    print(f"点击转换后的原始坐标: ({clicked_x:.2f}, {clicked_y:.2f})")

canvas.bind("<Button-1>", on_click)

# -------------------------------
# 7. 定义预测按钮回调（使用两输出模型，并进行归一化）
# -------------------------------
def predict_result():
    global clicked_x, clicked_y
    if clicked_x is None or clicked_y is None:
        messagebox.showerror("错误", "请先点击地图选择位置")
        return
    
    # 弹出对话框要求输入方向角（单位：度）
    direction_str = simpledialog.askstring("输入方向", "请输入方向角（单位：度）")
    try:
        direction_deg = float(direction_str)
    except Exception as e:
        messagebox.showerror("错误", "方向角必须为数字")
        return

    # 将方向角转换为弧度，并计算 sin 和 cos 分量
    direction_rad = math.radians(direction_deg)
    sin_view = math.sin(direction_rad)
    cos_view = math.cos(direction_rad)
    
    # 构造原始特征向量：[x, y, sin_view, cos_view]
    raw_feature = np.array([clicked_x, clicked_y, sin_view, cos_view]).reshape(1, -1)
    # 使用训练时 fit 的 scaler 进行归一化
    normalized_feature = scaler.transform(raw_feature)
    # 调整形状为 (samples, 4, 1) 以适应 Conv1D 模型输入格式
    feature = normalized_feature.reshape(1, 4, 1)
    
    # 使用模型进行预测，输出为两个概率：击杀概率和被击杀概率
    print(feature)
    feature = [[[-1.47571192]
  [-0.70386898]
  [-0.65683017]
  [-1.25153307]]]
    
    prediction = model.predict(feature)
    kill_prob = prediction[0][0]
    killed_prob = prediction[0][1]
    print(prediction)
    messagebox.showinfo("预测结果", f"击杀概率: {kill_prob:.2f}\n被击杀概率: {killed_prob:.2f}")

# -------------------------------
# 8. 添加预测按钮
# -------------------------------
predict_button = tk.Button(root, text="预测", command=predict_result)
predict_button.pack(pady=5)

root.mainloop()
