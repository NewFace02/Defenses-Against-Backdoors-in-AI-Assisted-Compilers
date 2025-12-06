const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

// Configuration
const SOURCE_DIR = path.join(__dirname, '..');
const DIST_DIR = path.join(__dirname, '..', 'dist');
const DOCS_DIR = path.join(__dirname, '..', 'docs');

// Create dist directory if it doesn't exist
if (!fs.existsSync(DIST_DIR)) {
  fs.mkdirSync(DIST_DIR, { recursive: true });
}

// HTML template
const htmlTemplate = (title, content) => `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title}</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      line-height: 1.6;
      max-width: 900px;
      margin: 0 auto;
      padding: 20px;
      color: #333;
    }
    h1, h2, h3, h4, h5, h6 {
      margin-top: 24px;
      margin-bottom: 16px;
      font-weight: 600;
      line-height: 1.25;
    }
    h1 { font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
    h2 { font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
    code {
      background-color: #f6f8fa;
      padding: 0.2em 0.4em;
      margin: 0;
      font-size: 85%;
      border-radius: 3px;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
    pre {
      background-color: #f6f8fa;
      padding: 16px;
      overflow: auto;
      font-size: 85%;
      line-height: 1.45;
      border-radius: 6px;
    }
    pre code {
      background-color: transparent;
      padding: 0;
    }
    blockquote {
      padding: 0 1em;
      color: #6a737d;
      border-left: 0.25em solid #dfe2e5;
      margin: 0;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 16px 0;
    }
    table th, table td {
      padding: 6px 13px;
      border: 1px solid #dfe2e5;
    }
    table tr:nth-child(2n) {
      background-color: #f6f8fa;
    }
    a {
      color: #0366d6;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    img {
      max-width: 100%;
      box-sizing: border-box;
    }
  </style>
</head>
<body>
${content}
</body>
</html>
`;

// Function to compile a single markdown file
function compileMarkdown(filePath, outputPath) {
  console.log(`Compiling ${filePath} -> ${outputPath}`);
  
  const markdown = fs.readFileSync(filePath, 'utf-8');
  const html = marked(markdown);
  const fileName = path.basename(filePath, '.md');
  const title = fileName.charAt(0).toUpperCase() + fileName.slice(1);
  
  const fullHtml = htmlTemplate(title, html);
  fs.writeFileSync(outputPath, fullHtml);
  
  console.log(`✓ Generated ${outputPath}`);
}

// Find all markdown files in root and docs directory
function findMarkdownFiles(dir) {
  const files = [];
  
  if (!fs.existsSync(dir)) {
    return files;
  }
  
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isFile() && item.endsWith('.md')) {
      files.push(fullPath);
    } else if (stat.isDirectory() && item !== 'node_modules' && item !== 'dist' && item !== '.git') {
      files.push(...findMarkdownFiles(fullPath));
    }
  }
  
  return files;
}

// Main build process
console.log('🚀 Starting Markdown to HTML compilation...\n');

// Find all markdown files
const markdownFiles = [
  ...findMarkdownFiles(SOURCE_DIR).filter(f => {
    const relative = path.relative(SOURCE_DIR, f);
    return !relative.includes('node_modules') && 
           !relative.includes('dist') && 
           !relative.startsWith('.');
  }),
  ...findMarkdownFiles(DOCS_DIR)
];

if (markdownFiles.length === 0) {
  console.log('⚠️  No markdown files found to compile.');
  console.log('   Create .md files in the root or docs/ directory.');
  process.exit(0);
}

// Compile each markdown file
for (const mdFile of markdownFiles) {
  const relativePath = path.relative(SOURCE_DIR, mdFile);
  const outputPath = path.join(DIST_DIR, relativePath.replace('.md', '.html'));
  
  // Create subdirectories if needed
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  compileMarkdown(mdFile, outputPath);
}

console.log(`\n✅ Successfully compiled ${markdownFiles.length} markdown file(s) to HTML!`);
console.log(`📁 Output directory: ${DIST_DIR}`);
