# 使用指南 (Usage Guide)

## 快速开始 (Quick Start)

### 1. 安装依赖

首次使用前，需要安装Node.js依赖包：

```bash
npm install
```

### 2. 编译Markdown文件

将Markdown文件编译为HTML：

```bash
npm run build
```

编译后的HTML文件将保存在 `dist/` 目录中。

## 编译命令详解

### 编译为HTML

```bash
npm run build
```

- 自动查找所有 `.md` 文件（根目录和 `docs/` 目录）
- 转换为格式美观的HTML文件
- 输出到 `dist/` 目录，保持原有目录结构

**示例输出：**
```
docs/introduction.md  →  dist/docs/introduction.html
README.md            →  dist/README.html
```

### 清理编译输出

```bash
npm run clean
```

删除 `dist/` 目录及所有编译生成的文件。

## 添加新文档

### 方法1：在根目录添加

直接在项目根目录创建 `.md` 文件：

```bash
echo "# My Document" > my-doc.md
npm run build
```

### 方法2：在docs目录添加

推荐将文档组织在 `docs/` 目录中：

```bash
echo "# Technical Details" > docs/technical.md
npm run build
```

## 文档组织建议

```
project/
├── README.md              # 项目主文档
├── docs/                  # 文档目录
│   ├── introduction.md    # 介绍
│   ├── defenses.md        # 防御方法
│   ├── research.md        # 研究进展
│   └── guide.md           # 使用指南
└── dist/                  # 编译输出（自动生成）
    ├── README.html
    └── docs/
        ├── introduction.html
        ├── defenses.html
        ├── research.html
        └── guide.html
```

## HTML样式特性

编译生成的HTML包含以下特性：

- ✅ 响应式设计，适配移动设备
- ✅ GitHub风格的Markdown渲染
- ✅ 代码语法高亮支持
- ✅ 表格、引用、列表等完整支持
- ✅ 中文字体优化

## 查看编译结果

### 在浏览器中打开

```bash
# Linux/Mac
open dist/README.html

# 或直接用浏览器打开
firefox dist/README.html
google-chrome dist/README.html
```

### 使用本地服务器（推荐）

如果需要预览多个HTML文件：

```bash
# 安装简单HTTP服务器
npm install -g http-server

# 在dist目录启动服务器
cd dist
http-server -p 8080

# 在浏览器访问 http://localhost:8080
```

## 常见问题

### Q: 为什么有些文件被编译了多次？

A: 这是findMarkdownFiles函数的重复检查问题，不影响最终结果，输出文件只会保留最后一次编译的版本。

### Q: 如何自定义HTML样式？

A: 编辑 `scripts/build.js` 中的 `htmlTemplate` 函数，修改CSS样式。

### Q: 支持哪些Markdown特性？

A: 支持GitHub Flavored Markdown (GFM)的所有特性：
- 标题、段落、列表
- 代码块和行内代码
- 表格
- 引用
- 链接和图片
- 任务列表
- 删除线、粗体、斜体等

### Q: 如何排除某些文件不被编译？

A: 编辑 `scripts/build.js`，在 `findMarkdownFiles` 函数中添加过滤条件。

## 故障排除

### 编译失败

1. 确保已安装依赖：`npm install`
2. 检查Node.js版本：`node --version`（需要v14+）
3. 查看错误信息，检查Markdown语法

### 找不到文件

确保：
- `.md` 文件在根目录或 `docs/` 目录
- 文件名以 `.md` 结尾
- 文件不在 `node_modules/` 或 `.git/` 目录中

## 工作流程示例

### 典型的文档编写流程

```bash
# 1. 创建或编辑Markdown文件
vim docs/new-feature.md

# 2. 编译为HTML
npm run build

# 3. 在浏览器中预览
open dist/docs/new-feature.html

# 4. 如需修改，重复步骤1-3

# 5. 完成后提交到Git
git add docs/new-feature.md
git commit -m "Add new feature documentation"
```

## 高级用法

### 批量处理

编译系统会自动处理所有Markdown文件，无需单独指定。

### 监视模式（未实现）

如需自动重新编译，可以使用 `nodemon` 等工具：

```bash
npm install -g nodemon
nodemon --watch docs --watch README.md --exec "npm run build"
```

### 集成到CI/CD

在CI/CD流程中自动编译文档：

```yaml
# .github/workflows/docs.yml
- name: Build documentation
  run: |
    npm install
    npm run build
```

## 下一步

- 阅读 [防御方法](defenses.md) 了解具体的防御策略
- 查看 [项目简介](introduction.md) 了解研究背景
- 浏览 `References/` 目录中的学术论文
