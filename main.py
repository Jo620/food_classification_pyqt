import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QProgressBar, QTableWidget, QTableWidgetItem, 
                             QSplitter, QGroupBox, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QColor, QLinearGradient
import torch

from model_loader import load_model
from predictor import FoodPredictor


class PredictThread(QThread):
    """预测线程，避免界面卡顿"""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, predictor, image_path):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path
    
    def run(self):
        try:
            results = self.predictor.predict(self.image_path, top_k=5)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class FoodClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 配置参数
        self.model_path = 'models/best_model.pth'  # 修改为你的模型路径
        self.num_classes = 202
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化变量
        self.current_image_path = None
        self.predict_count = 0
        self.model = None
        self.predictor = None
        
        # 初始化界面
        self.init_ui()
        
        # 加载模型
        self.load_model()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('🍽️ GlobalFood202 食物分类识别系统')
        self.setGeometry(100, 100, 1400, 800)
        
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
            }
            QWidget {
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial;
            }
            QGroupBox {
                background-color: white;
                border-radius: 15px;
                margin-top: 10px;
                font-weight: bold;
                padding: 15px;
            }
            QGroupBox::title {
                color: #667eea;
                font-size: 18px;
                padding: 5px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #764ba2, stop:1 #667eea);
            }
            QPushButton:pressed {
                background: #5568d3;
            }
            QPushButton:disabled {
                background: #cccccc;
            }
            QLabel {
                color: #333;
            }
            QProgressBar {
                border: 2px solid #667eea;
                border-radius: 8px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 6px;
            }
            QTableWidget {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #667eea;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #b8c5ff;
            }
        """)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # 标题栏
        self.create_header(main_layout)
        
        # 统计信息栏
        self.create_stats_bar(main_layout)
        
        # 创建分割器（左右布局）
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：图像上传和预览区域
        left_widget = self.create_left_panel()
        splitter.addWidget(left_widget)
        
        # 右侧：预测结果区域
        right_widget = self.create_right_panel()
        splitter.addWidget(right_widget)
        
        splitter.setSizes([600, 700])
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.statusBar().showMessage('✅ 系统就绪')
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: rgba(255, 255, 255, 0.9);
                color: #333;
                font-weight: bold;
                border-top: 2px solid #667eea;
            }
        """)
    
    def create_header(self, parent_layout):
        """创建标题栏"""
        header_widget = QWidget()
        header_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 15px;
                padding: 10px;
            }
        """)
        header_layout = QVBoxLayout(header_widget)
        
        # 主标题
        title_label = QLabel('🍽️ GlobalFood202 食物分类识别系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: white;
            padding: 10px;
        """)
        header_layout.addWidget(title_label)
        
        # 副标题
        subtitle_label = QLabel('基于 Swin Transformer + FPN 注意力融合网络')
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("""
            font-size: 16px;
            color: rgba(255, 255, 255, 0.9);
            padding: 5px;
        """)
        header_layout.addWidget(subtitle_label)
        
        parent_layout.addWidget(header_widget)
    
    def create_stats_bar(self, parent_layout):
        """创建统计信息栏"""
        stats_widget = QWidget()
        stats_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 15px;
                padding: 10px;
            }
        """)
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setSpacing(20)
        
        # 统计项样式
        stat_style = """
            QLabel {
                color: white;
                padding: 8px 20px;
                background-color: rgba(255, 255, 255, 0.2);
                border-radius: 10px;
            }
        """
        
        # 类别数
        self.class_count_label = QLabel(f'📊 类别数: {self.num_classes}')
        self.class_count_label.setStyleSheet(stat_style)
        self.class_count_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.class_count_label)
        
        # 设备信息
        device_name = "GPU" if self.device.type == "cuda" else "CPU"
        self.device_label = QLabel(f'🖥️  设备: {device_name}')
        self.device_label.setStyleSheet(stat_style)
        self.device_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.device_label)
        
        # 识别次数
        self.count_label = QLabel(f'🔢 识别次数: {self.predict_count}')
        self.count_label.setStyleSheet(stat_style)
        self.count_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.count_label)
        
        parent_layout.addWidget(stats_widget)
    
    def create_left_panel(self):
        """创建左侧面板（图像上传区域）"""
        left_group = QGroupBox('📸 图像上传')
        left_layout = QVBoxLayout(left_group)
        left_layout.setSpacing(15)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton('📁 选择图片')
        self.upload_btn.setMinimumHeight(50)
        self.upload_btn.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_btn)
        
        self.predict_btn = QPushButton('🔍 开始识别')
        self.predict_btn.setMinimumHeight(50)
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self.start_prediction)
        button_layout.addWidget(self.predict_btn)
        
        left_layout.addLayout(button_layout)
        
        # 图像预览区域
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.Box)
        preview_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9ff;
                border: 3px dashed #667eea;
                border-radius: 15px;
            }
        """)
        preview_layout = QVBoxLayout(preview_frame)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 500)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: #999;
                font-size: 18px;
            }
        """)
        self.image_label.setText('📷\n\n请上传食物图片')
        preview_layout.addWidget(self.image_label)
        
        left_layout.addWidget(preview_frame)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(30)
        left_layout.addWidget(self.progress_bar)
        
        return left_group
    
    def create_right_panel(self):
        """创建右侧面板（预测结果区域）"""
        right_group = QGroupBox('🏆 识别结果')
        right_layout = QVBoxLayout(right_group)
        
        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(['排名', '食物名称', '置信度'])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setColumnWidth(0, 80)
        self.result_table.setColumnWidth(1, 300)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.verticalHeader().setVisible(False)
        
        # 设置表格字体
        font = QFont()
        font.setPointSize(11)
        self.result_table.setFont(font)
        
        right_layout.addWidget(self.result_table)
        
        # 清空按钮
        clear_btn = QPushButton('🗑️ 清空结果')
        clear_btn.clicked.connect(self.clear_results)
        right_layout.addWidget(clear_btn)
        
        return right_group
    
    def load_model(self):
        """加载模型"""
        try:
            self.statusBar().showMessage('🔄 正在加载模型...')
            QApplication.processEvents()
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            self.model = load_model(self.model_path, self.num_classes, self.device)
            self.predictor = FoodPredictor(self.model, device=self.device)
            
            self.statusBar().showMessage('✅ 模型加载成功！')
            QMessageBox.information(self, '成功', '✅ 模型加载成功！\n可以开始识别了。')
            
        except Exception as e:
            error_msg = f'❌ 模型加载失败:\n{str(e)}'
            self.statusBar().showMessage('❌ 模型加载失败')
            QMessageBox.critical(self, '错误', error_msg)
            print(error_msg)
    
    def upload_image(self):
        """上传图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择食物图片',
            '',
            'Images (*.png *.jpg *.jpeg *.bmp *.gif)'
        )
        
        if file_path:
            self.current_image_path = file_path
            
            # 显示图片
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            # 启用识别按钮
            self.predict_btn.setEnabled(True)
            self.statusBar().showMessage(f'✅ 已加载图片: {os.path.basename(file_path)}')
    
    def start_prediction(self):
        """开始预测"""
        if not self.current_image_path or not self.predictor:
            return
        
        # 禁用按钮，显示进度条
        self.predict_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.statusBar().showMessage('🔍 正在识别...')
        
        # 创建预测线程
        self.predict_thread = PredictThread(self.predictor, self.current_image_path)
        self.predict_thread.finished.connect(self.on_prediction_finished)
        self.predict_thread.error.connect(self.on_prediction_error)
        self.predict_thread.start()
    
    def on_prediction_finished(self, results):
        """预测完成的回调"""
        # 隐藏进度条，启用按钮
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        
        # 显示结果
        self.display_results(results)
        
        # 更新统计
        self.predict_count += 1
        self.count_label.setText(f'🔢 识别次数: {self.predict_count}')
        
        self.statusBar().showMessage('✅ 识别完成！')
    
    def on_prediction_error(self, error_msg):
        """预测错误的回调"""
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        
        QMessageBox.critical(self, '预测错误', f'❌ 预测失败:\n{error_msg}')
        self.statusBar().showMessage('❌ 预测失败')
    
    def display_results(self, results):
        """显示预测结果"""
        self.result_table.setRowCount(len(results))
        
        # 排名图标
        rank_icons = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        
        for i, result in enumerate(results):
            # 排名
            rank_item = QTableWidgetItem(f"{rank_icons[i]} {i+1}")
            rank_item.setTextAlignment(Qt.AlignCenter)
            font = QFont()
            font.setPointSize(12)
            font.setBold(True)
            rank_item.setFont(font)
            
            # 第一名高亮
            if i == 0:
                rank_item.setBackground(QColor(255, 215, 0, 100))
            
            self.result_table.setItem(i, 0, rank_item)
            
            # 类别名称
            name_item = QTableWidgetItem(result['class_name'])
            name_item.setFont(font)
            if i == 0:
                name_item.setBackground(QColor(255, 215, 0, 100))
            self.result_table.setItem(i, 1, name_item)
            
            # 置信度
            confidence = result['probability'] * 100
            conf_item = QTableWidgetItem(f"{confidence:.2f}%")
            conf_item.setTextAlignment(Qt.AlignCenter)
            conf_item.setFont(font)
            
            # 根据置信度设置颜色
            if confidence >= 80:
                conf_item.setForeground(QColor(0, 150, 0))
            elif confidence >= 50:
                conf_item.setForeground(QColor(200, 100, 0))
            else:
                conf_item.setForeground(QColor(150, 0, 0))
            
            if i == 0:
                conf_item.setBackground(QColor(255, 215, 0, 100))
            
            self.result_table.setItem(i, 2, conf_item)
        
        # 调整行高
        for i in range(len(results)):
            self.result_table.setRowHeight(i, 50)
    
    def clear_results(self):
        """清空结果"""
        self.result_table.setRowCount(0)
        self.image_label.clear()
        self.image_label.setText('📷\n\n请上传食物图片')
        self.current_image_path = None
        self.predict_btn.setEnabled(False)
        self.statusBar().showMessage('✅ 已清空结果')


def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = FoodClassificationApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
