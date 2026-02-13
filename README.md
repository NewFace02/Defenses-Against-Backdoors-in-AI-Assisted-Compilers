# Defenses Against Backdoors in AI-Assisted Compilers

This repository contains research and references on defending against backdoor attacks in AI-assisted compilers and code generation systems.

## 📚 References

The `References/` directory contains academic papers and resources on:
- Backdoor attacks on compilers
- Stego attacks on open-source Large Language Models (LLMs)

## 📝 编译Markdown文件 (Compiling Markdown Files)

本项目支持将Markdown文件编译为HTML格式。

### 前置要求 (Prerequisites)

- Node.js (v14 或更高版本)
- npm (Node Package Manager)

### 安装依赖 (Installation)

```bash
npm install
```

### 编译MD为HTML (Compile MD to HTML)

```bash
npm run build
```

这将把所有的 `.md` 文件编译为 HTML 文件，输出到 `dist/` 目录。

### 查看编译结果 (View Results)

编译后的文件位于 `dist/` 目录中，可以直接在浏览器中打开HTML文件。

## 🛠️ Development

### Available Scripts

- `npm run build` - 编译Markdown为HTML
- `npm run clean` - 清理编译输出

## 📖 Documentation Structure

建议的文档结构：

```
docs/
├── introduction.md      # 项目介绍
├── background.md        # 背景知识
├── defenses.md          # 防御方法
├── research.md          # 研究进展
└── references.md        # 参考文献
```

## 🔒 Security

本项目专注于研究AI辅助编译器中的后门攻击防御机制。如发现安全问题，请负责任地披露。

## 📄 License

请参考各参考文献的原始许可证。

## 🤝 Contributing

欢迎贡献！请确保：
1. 添加的文档使用Markdown格式
2. 运行 `npm run build` 确保文档可以正确编译
3. 更新相关的目录结构

---

**Note**: This repository is for research and educational purposes.
