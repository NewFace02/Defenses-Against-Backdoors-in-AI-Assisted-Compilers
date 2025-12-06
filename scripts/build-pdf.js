const fs = require('fs');
const path = require('path');
const markdownpdf = require('markdown-pdf');

// Configuration
const SOURCE_DIR = path.join(__dirname, '..');
const DIST_DIR = path.join(__dirname, '..', 'dist');
const DOCS_DIR = path.join(__dirname, '..', 'docs');

// Create dist directory if it doesn't exist
if (!fs.existsSync(DIST_DIR)) {
  fs.mkdirSync(DIST_DIR, { recursive: true });
}

// Find all markdown files
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
console.log('🚀 Starting Markdown to PDF compilation...\n');

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
let completed = 0;
let errors = 0;

for (const mdFile of markdownFiles) {
  const relativePath = path.relative(SOURCE_DIR, mdFile);
  const outputPath = path.join(DIST_DIR, relativePath.replace('.md', '.pdf'));
  
  // Create subdirectories if needed
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  console.log(`Compiling ${mdFile} -> ${outputPath}`);
  
  markdownpdf()
    .from(mdFile)
    .to(outputPath, () => {
      completed++;
      console.log(`✓ Generated ${outputPath}`);
      
      if (completed + errors === markdownFiles.length) {
        console.log(`\n✅ Successfully compiled ${completed} markdown file(s) to PDF!`);
        if (errors > 0) {
          console.log(`⚠️  ${errors} file(s) failed to compile.`);
        }
        console.log(`📁 Output directory: ${DIST_DIR}`);
      }
    });
}

// Handle errors
process.on('uncaughtException', (err) => {
  errors++;
  console.error('Error:', err.message);
});
