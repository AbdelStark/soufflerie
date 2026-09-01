# Security policy

## Supported versions

Before the first tagged release, security fixes target the latest commit on
`main`. After releases begin, only the newest published minor release and
`main` are supported unless a release note explicitly says otherwise. There are
currently no published release lines with a separate support window.

## Report a vulnerability privately

Do not open a public issue or pull request for a suspected vulnerability. Use
GitHub's private reporting flow:

<https://github.com/AbdelStark/soufflerie/security/advisories/new>

Include the affected revision/version, impact, prerequisites, minimal
reproduction, and any suggested mitigation. Use synthetic data and credentials;
never send a real token, private dataset, or unnecessary personal information.
If the private form is unavailable, open a public issue containing no sensitive
details and ask `@AbdelStark` to restore a private contact path.

The project has one maintainer and no 24/7 response team or bug-bounty program.
The targets are a human acknowledgment within 7 calendar days and an initial
triage update within 14 calendar days. These are best-effort targets, not a
service-level agreement. Remediation and coordinated disclosure timing depend
on severity, reproducibility, upstream dependencies, and release safety; the
reporter will receive status updates through the private advisory while work is
active.

Please keep the report private until a fix and disclosure plan are agreed. The
maintainer will credit reporters who want attribution and will preserve
anonymity when requested, subject to platform and legal constraints.

## Security scope

Reports are especially useful for:

- artifact traversal, unsafe deserialization, digest/lineage bypass, or
  allocation-limit bypass;
- secret disclosure through configuration, distributions, logs, errors, health,
  reports, or repository history;
- public service admission, budget, CORS, proxy, or readiness bypass;
- dependency or build/release provenance compromise; and
- vulnerabilities in the supported local package or documented deployment.

Model-quality disagreement, unsupported scientific domains, ordinary bugs, and
feature requests are not security vulnerabilities unless they cross a stated
integrity or safety boundary. Soufflerie is educational research software, not
an engineering-certification or safety-critical system.

## Disclosure and fixes

Security fixes may break compatibility when exploitation risk outweighs the
normal deprecation window. The advisory and changelog will describe affected
versions, impact, mitigation, fixed versions, and any artifact invalidation
without publishing exploit details before users can update. Dependencies are
coordinated with their upstream maintainers when appropriate.
