# IEEE HealthCom paper page

A static academic project page for **Dual-Track Longitudinal Modeling of Diabetes
Risk: Horizon-Specific Screening and Intervention-Safe Scoring**. It presents
the paper's reported results, two original figures, and links to the research
implementation and its validation notes. Paper links are placeholders until
the official IEEE publication URL is available; no paper PDF is distributed.

GitHub Pages address (after deployment):
<https://pitikorn32.github.io/dual-track-longitudinal-diabetes-risk/>

## Preview and check

Use Node.js 24 and Python 3. From this directory:

```bash
npm ci
npx playwright install chromium
npm test
npm run preview
```

Open <http://127.0.0.1:4173/> for the preview. The site uses HTML, CSS, and
browser JavaScript; it needs no application server or patient data. Inter is
self-hosted with its SIL Open Font License in `assets/fonts/OFL.txt`.

Browser checks cover desktop and mobile layouts, paper-reported results,
horizon selection, figure expansion and keyboard focus, citation copying and
its fallback, public assets and BibTeX downloads under a subdirectory URL, accessibility,
and useful content with JavaScript disabled.

`npm run build` writes `dist/` from an explicit list of page files and curated
assets. Tests, dependencies, source code, and research artifacts are excluded
from the published directory.

## GitHub Pages

In this repository's **Settings → Pages → Build and deployment**, select
**GitHub Actions** as the source. The
[Paper page workflow](../.github/workflows/paper-page.yml) then:

1. Builds and checks the page on pull requests that affect the website.
2. Builds, checks, and deploys website changes pushed to `main`.
3. Supports a manual run from the Actions tab on `main`.

Deployment publishes only `website/dist/` to the `github-pages` environment.
Pull requests have read-only repository permissions and cannot deploy. The
research workflow remains separate from this website workflow.

See [GitHub's Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Updating research content

The HTML results table is the source for the horizon explorer and its chart.
Keep its numbers labeled as paper-reported results; current reruns and version
differences belong in the linked research validation notes. Reported differences
and lifts use the paper's original precision rather than subtraction of rounded
table values.

When the official IEEE publication URL becomes available, replace the `href`
on the three `data-paper-link` anchors in `index.html` and update their
coming-soon labels and the availability notice. Add a DOI or publisher URL to
the citation metadata only when the official details are known.

The reproduced figures retain the paper's **© 2026 IEEE** notice.
The repository's MIT software license does not replace that notice. Figures
are reproduced unchanged and attributed on the page. The overview illustration
is conceptual; illustrative diagram scores are not numerical study results.
