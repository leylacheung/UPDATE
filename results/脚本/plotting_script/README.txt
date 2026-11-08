本次实际绘图脚本的便携版：绘图逻辑保持不变，仅输入输出路径改为命令行参数，并自动识别CSV或制表符分隔。
推荐Python 3.12。解压后进入目录运行：
python -m pip install -r requirements.txt
python plot_metrics.py --input test_results.csv --output figures

可用完整汇总表替代test_results.csv，自动筛选split=test。脚本针对本次完整矩阵：116条测试结果、10个条件、4种模型、每组5折，不支持任意子集。
输出：总览PNG/SVG、各条件PNG、12页PDF及打包文件。
R²直接读取输入表，本脚本不重算R²，也不训练模型。
