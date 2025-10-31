# config.py
# 训练配置文件

import os

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

# 数据配置
DATA_CONFIG = {
    'en_file': 'MDN_Web_Docs.en-zh_CN.en',
    'zh_file': 'MDN_Web_Docs.en-zh_CN.zh_CN',
    'data_dir': 'data',
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'batch_size': 32,
    'max_vocab_size': 50000
}

# 模型配置
MODEL_CONFIG = {
    'layers': 6,
    'd_model': 512,
    'd_ff': 2048,
    'h_num': 8,
    'dropout': 0.1,
    'max_length': 60
}

# 训练配置
TRAINING_CONFIG = {
    'epochs': 300,
    'learning_rate': 0.0001,
    'warmup_steps': 4000,
    'grad_clip': 1.0,
    'label_smoothing': 0.1,
    'patience': 10,
    'resume_training': True,
    'save_every': 10,  # 每10个epoch保存一次检查点
    'print_every': 100  # 每100个batch打印一次训练信息
}

# 路径配置
PATH_CONFIG = {
    'model_path': os.path.join(project_root, 'model.pt'),
    'best_model_path': os.path.join(project_root, 'best_model.pt'),
    'checkpoint_dir': os.path.join(project_root, 'checkpoints'),
    'log_file': os.path.join(project_root, 'training.log')
}

# 设备配置
DEVICE_CONFIG = {
    'use_cuda': True,
    'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
}

def get_config():
    """获取完整配置"""
    config = {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'paths': PATH_CONFIG,
        'device': DEVICE_CONFIG
    }
    return config

if __name__ == "__main__":
    config = get_config()
    print("训练配置:")
    for category, settings in config.items():
        print(f"\n{category.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
