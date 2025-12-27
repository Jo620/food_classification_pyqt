import torch
from torchvision import transforms
from PIL import Image
import json

class FoodPredictor:
    def __init__(self, model, class_names_path='class_names.json', device='cpu'):
        """
        初始化预测器
        
        Args:
            model: 加载好的模型
            class_names_path: 类别名称JSON文件路径
            device: 计算设备
        """
        self.model = model
        self.device = device
        
        # 加载类别名称
        try:
            with open(class_names_path, 'r', encoding='utf-8') as f:
                self.class_names = json.load(f)
            print(f"✅ 成功加载 {len(self.class_names)} 个类别")
        except:
            self.class_names = [f"食物类别_{i}" for i in range(202)]
            print("⚠️ 使用默认类别名称")
        
        # 图像预处理（与训练时的验证集一致）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=5):
        """
        预测图像类别
        
        Args:
            image_path: 图像文件路径
            top_k: 返回Top-K个预测结果
        
        Returns:
            list: 包含 (类别名称, 类别ID, 概率) 的列表
        """
        try:
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 获取Top-K预测
                top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))
            
            # 整理结果
            results = []
            for i in range(len(top_probs[0])):
                class_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = self.class_names[class_idx]
                
                results.append({
                    'class_name': class_name,
                    'class_id': class_idx,
                    'probability': prob
                })
            
            return results
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            raise
