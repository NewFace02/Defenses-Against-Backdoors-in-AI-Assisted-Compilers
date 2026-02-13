const fs = require('fs');
const path = require('path');

// Configuration
const DIST_DIR = path.join(__dirname, '..', 'dist');

// Function to recursively delete directory
function deleteDirectory(dirPath) {
  if (fs.existsSync(dirPath)) {
    fs.rmSync(dirPath, { recursive: true, force: true });
  }
}

console.log('🧹 Cleaning build output...\n');

if (fs.existsSync(DIST_DIR)) {
  deleteDirectory(DIST_DIR);
  console.log(`✓ Removed ${DIST_DIR}`);
  console.log('\n✅ Clean completed!');
} else {
  console.log('⚠️  No dist directory found. Nothing to clean.');
}
