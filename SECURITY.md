# Security

## Reporting a vulnerability

**Preferred:** use GitHub's [Private Vulnerability Reporting](https://github.com/airlockrun/goai/security/advisories/new).

**Fallback:** email `security@airlock.run`.

**Don't** open a public issue for vulnerabilities.

## What's a vulnerability in a library

- A flaw in goai that, when used as documented, makes the dependent application vulnerable (memory corruption, panic on attacker-controlled input, parser confusion, prompt-injection enabling defect, credential leakage, etc.).
- A defect in security-relevant code (HTTP transport, model authentication, response parsing).

## What's not

- Bugs without a security impact — open a regular issue instead.
- Vulnerabilities in libraries that goai depends on — report to the upstream first; we'll bump once they patch.
- Bugs that are inherited from the upstream [Vercel AI SDK](https://github.com/vercel/ai) and exist in their TypeScript implementation as well — please **also** report to vercel/ai. The fix will likely flow into goai once they ship.
- Misuse: the dependent application using goai in a way the docs warn against.

## What you can expect

- **Acknowledgment within 72 hours.**
- **Triage within 7 days.**
- **Fix targeted within 30 days for High/Critical, 90 days for Low/Medium.**
- Credit in the security advisory unless you ask to remain anonymous.

## Safe harbor

Good-faith research won't trigger legal action. Don't disclose publicly before we've patched (or 90 days, whichever first). Don't demand payment as a condition of disclosure.
