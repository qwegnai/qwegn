# Security Policy

## Reporting a vulnerability

Please report security issues privately. Do not open a public issue or pull request for them.

Use GitHub's private vulnerability reporting on this repository: open the **Security** tab and choose **Report a vulnerability**. This sends the report only to the maintainer.

Include:

- the affected version or commit, and the macOS and Xcode versions;
- the steps or input that reproduce the problem;
- what you expected to happen and what happened instead.

You can expect an acknowledgement within 7 days. The maintainer will confirm whether the report is accepted, work on a fix, and credit the reporter in the release notes unless you ask not to be named.

## Scope

In scope: the `qwg` command line tool, the `qwg serve` HTTP server, the `qwg-menu` app, pack and checkpoint parsing, and the Python golden generators.

Out of scope: vulnerabilities in macOS, Xcode, Metal, or third-party models and packages. Report those to their own maintainers.

## Supported versions

Only the latest release on `main` receives security fixes.
