# 运行方法

### 前置条件
请确保你的系统已安装以下工具：
- **uv**： python 包管理工具
- **Jupyter**： 交互式编程环境

### 配置环境变量
1. 在项目根目录下创建名为.env的文件
2. 向该文件中添加以下内容（替换xxx为你的实际密钥）：
3. 若没有 API 密钥，可前往 https://www.deepseek.com/api-docs 申请

```env
# DeepSeek API 密钥配置
DEEPSEEK_API_KEY=xxx
