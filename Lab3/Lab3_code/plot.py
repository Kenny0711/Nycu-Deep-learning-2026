import subprocess
import matplotlib.pyplot as plt

# 1. 參數設定
total_iter = 10
sweet_spots = list(range(1, 11))  # X 軸：1 到 10
func = 'cosine'                   # 只測試 cosine 函數

fid_scores = []                   # 用來儲存 10 次的 FID 分數

print(f"🚀 開始測試 {func} 函數，Sweet Spot 從 1 到 {total_iter}...")

# 2. 跑迴圈測試 1 到 10
for t in sweet_spots:
    print(f"⏳ 正在測試 t={t:<2}...", end=" ", flush=True)
    
    # 步驟 A: 執行 inpainting.py 生圖 (假設會將結果覆蓋存於 test_results)
    cmd_inpaint = f"python inpainting.py --mask-func {func} --total-iter {total_iter} --sweet-spot {t}"
    subprocess.run(cmd_inpaint, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 步驟 B: 執行 fid_score_gpu.py 算分數
    # 注意：請確認你的路徑 --predicted-path 和 --gtcsv-path 正確無誤
    cmd_fid = cmd_fid = "python faster-pytorch-fid/fid_score_gpu.py --predicted-path C:/Lab3/Lab3_code/test_results --num-workers 0 --gtcsv-path faster-pytorch-fid/test_gt.csv"
    output = subprocess.getoutput(cmd_fid)
    
    # 步驟 C: 擷取 FID 分數
    try:
        fid_str = output.split('FID:')[1].strip()
        score = float(fid_str)
        fid_scores.append(score)
        print(f"✅ FID: {score:.4f}")
    except IndexError:
        print("❌ 計算失敗 (找不到 FID 輸出)")
        print(f"👉 系統真實錯誤訊息：\n{output}")
        fid_scores.append(None)

# 3. 開始畫圖
print("\n📊 數據收集完畢，正在繪製折線圖...")
plt.figure(figsize=(8, 5))

# 畫出折線圖 (加上 marker='o' 可以畫出明顯的圓點)
plt.plot(sweet_spots, fid_scores, marker='o', label='cosine', color='#ff7f0e', linewidth=2)

# 4. 設定圖表美觀格式
plt.title(f'Effect of Sweet Spot on FID (T={total_iter}, Func={func})', fontsize=14)
plt.xlabel('Sweet Spot (t)', fontsize=12)
plt.ylabel('FID Score', fontsize=12)
plt.xticks(sweet_spots)               # 強制 X 軸顯示整數 1~10
plt.grid(True, linestyle='--', alpha=0.6) # 加上虛線網格，方便對齊看分數
plt.legend()

# 存檔
save_name = 'fid_sweet_spot_cosine.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
print(f"🎉 畫圖完成！圖表已儲存為 '{save_name}'")