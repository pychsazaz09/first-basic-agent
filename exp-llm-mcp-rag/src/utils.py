
def logTile(message: str):
    total_length = 80
    msg_len = len(message)
    # 总填充长度，max防止超长文字出现负数
    padding = max(0, total_length - msg_len - 4)
    # 左右分割填充，floor左、ceil右，奇数余量给右边
    left_pad = padding // 2
    right_pad = (padding + 1) // 2
    # 拼接：==== 文字 ==== 居中格式
    padded_msg = f"{'=' * left_pad} {message} {'=' * right_pad}"
    # 亮青色 + 加粗 打印
    print(padded_msg)