from ultralytics import YOLO

# 加载训练好的模型
model = YOLO(r'F:\VsCodeProject\dataset\Guesture_yolov8\checkpoint\medium\model.pt')  # 使用你训练好的模型权重文件

# 使用摄像头进行实时检测
model.predict(
    source=0,                      # 0 表示使用本地摄像头。你也可以使用视频文件路径或网络摄像头地址
    show=True,                     # 是否显示实时检测窗口
    conf=0.5,                      # 置信度阈值，低于此值的框会被忽略
    imgsz=640,                     # 输入图像大小
    device='0',                    # 使用的GPU设备
    save=True,                     # 是否保存检测结果到文件
    save_txt=False,                # 不保存为 TXT 文件
    save_crop=False,               # 是否保存检测框内的裁剪结果
    line_thickness=2,              # 框的线条粗细
)
