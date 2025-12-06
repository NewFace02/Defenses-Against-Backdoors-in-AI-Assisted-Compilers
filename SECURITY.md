# Security Policy

## Known Vulnerabilities

### markdown-pdf Dependency

The `markdown-pdf` package (v10.0.0) used for PDF compilation has a known security vulnerability:

- **Issue**: Local file read via server-side cross-site scripting (XSS)
- **Affected Versions**: <= 11.0.0
- **Patched Version**: Not available
- **CVSS**: Medium severity

### Mitigation

1. **Recommended**: Use HTML compilation instead of PDF compilation
   ```bash
   npm run build  # Safe - uses 'marked' package
   ```

2. If PDF compilation is required:
   - Only use in trusted, local environments
   - Do not expose the build process to untrusted input
   - Do not run on public-facing servers
   - Consider using alternative PDF generation tools

### Safe Usage

The HTML compilation feature (`npm run build`) uses the `marked` package which is actively maintained and does not have known critical vulnerabilities. This is the recommended method for compiling documentation.

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
