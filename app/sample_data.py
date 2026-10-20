"""Built-in sample corpus used for first-run seeding and offline evaluation."""

SAMPLE_COMPANY_HANDBOOK = """# ACME CORP — ENGINEERING & OPERATIONS HANDBOOK (2026)

## 1. Remote Work & Core Hours
Acme Corp operates on a hybrid-first engineering model. Core synchronous collaboration hours are 13:00 UTC to 17:00 UTC Monday through Friday.
All full-time employees receive an annual $2,500 Home Office Stipend and $1,200 Learning & Conference budget.

## 2. Paid Time Off (PTO) & Sabbatical Policy
Employees accrue 25 days of Paid Time Off (PTO) per calendar year, plus 12 company holidays.
After 4 years of continuous tenure, engineers qualify for a 4-week paid sabbatical.

## 3. Production Deployment & Incident Response
- **Deployment Freeze:** No routine production deployments on Fridays after 15:00 UTC without VP of Engineering sign-off.
- **SEV-1 SLA:** Critical customer-facing outages require initial on-call acknowledgement via PagerDuty within 5 minutes and a public status page update within 15 minutes.
- **Postmortems:** Blameless postmortems must be published in the `/incidents` repository within 48 hours of resolution.

## 4. Security & Credential Hygiene
All API keys and production secrets must be encrypted at rest using AES-256-GCM or stored in Vault. Rotating service tokens every 90 days is mandatory.
"""
