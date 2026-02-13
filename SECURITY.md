# Security Policy

## Security Status

✅ **All dependencies are secure**

This project uses only the `marked` package (v11.1.1) for Markdown compilation, which is actively maintained and has no known critical vulnerabilities.

## Previous Vulnerability (Resolved)

Earlier versions of this project included the `markdown-pdf` package which had a known XSS vulnerability with no available patch. This dependency has been **removed** to ensure the security of the project.

### What Changed

- ❌ Removed `markdown-pdf` dependency (vulnerable)
- ❌ Removed PDF compilation feature (`npm run build:pdf`)
- ✅ Kept HTML compilation feature (`npm run build`) - fully secure

## Safe Usage

The HTML compilation feature is completely safe to use:

```bash
npm install  # Installs only secure dependencies
npm run build  # Safe - uses 'marked' package
```

The generated HTML files use client-side rendering only and do not execute any server-side code.

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public issue
2. Contact the repository maintainers privately
3. Provide details about the vulnerability
4. Allow time for a fix before public disclosure

## Security Best Practices

When using this project:

1. Always run `npm audit` after installing dependencies
2. Keep dependencies updated when possible
3. Use HTML compilation for production use
4. Only use PDF compilation in isolated, trusted environments
5. Review the generated output before sharing

## Updates

We will update this security policy as vulnerabilities are discovered and addressed.

Last updated: 2025-12-06
