# Delivery verification

Executed on 26 October 2026 in the local macOS development environment.

## Passed

- Frontend: `npm run build` — TypeScript check and Vite production build succeeded.
- Backend: imported and started Flask successfully under Python 3.12 with TensorFlow 2.20.
- API suite: `python -m pytest backend/tests -q` — **13 passed**.
- Tests cover health, 43 IDs, fixed class mapping consistency with the frontend, missing uploads, corrupt images, request-size rejection, missing-model HTTP 503, absent metrics and matrix, disabled web training, missing sample routes, shared RGB/resize preprocessing, actual Keras inference, model contract/checksum and mapping checks, real artifact reading, and PPM preview conversion.
- A separate engineering smoke run trained for one epoch on **synthetic random images**, in a copy under `work/`. It verified checkpoint saving/reloading, embedded preprocessing, the model contract, API inference, all exported JSON files, a 43 × 43 confusion matrix whose count matched validation samples, and 43 image sample exports. **These images are not traffic signs. This check provides no recognition accuracy evidence. No generated model or metrics from it are bundled or exposed in the delivered dashboard.**
- HTTP health endpoint also checked through the running Vite proxy.
- Browser: opened all ten navigation sections; checked Overview CTA, class search, detail modal, absent dataset/results states, expanded convolution explanation, training command copy feedback, quiz answer feedback, documentation and editable student-detail state.
- Browser upload: selected a PNG, observed preview, submitted prediction and observed the actual backend’s missing-model error; removed the image successfully.
- Responsive checks: desktop screenshot; 390 px mobile menu and studio; 768 px tablet. Dataset page DOM width equalled viewport width at 390 and 768 px (no horizontal document overflow).
- Modal keyboard behavior includes Escape, focus containment and focus restoration. Reduced-motion CSS is provided. A full accessibility audit was not performed.

## Limits and warnings

- No GTSRB dataset or trained traffic-sign weights were supplied. No full GTSRB training, recognition-accuracy measurement or independent official-test evaluation was performed.
- Browser result charts/history and populated evaluation curves/matrix require real training artifacts. Their populated browser states were not verified against a real GTSRB run. Actual inference/API math and artifact shapes were verified in isolated engineering tests.
- Windows setup commands are documented, but this environment is macOS; native Windows installation was not executed.
- The Vite build reports a non-fatal JavaScript chunk-size advisory (about 607 kB minified, 175 kB gzip). Keras emitted NumPy deprecation warnings during tests; all tests passed.
- Web training is intentionally disabled. The working CLI is the supported training path.
- Original verification covered local development. Vercel deployment checks are recorded separately below.
- Full browser accessibility, GPU training, production concurrency, all operating systems and unusual image decoder edge cases were not tested.


## Vercel production deployment — 26 October 2026

- URL: https://signallab-traffic-sign-recognition.vercel.app
- Deployment: dpl_8Fg3i3BUDBTTE4k1XhUxW7pDL1dX — READY.
- Vercel frontend build passed; the Python API dependencies are explicitly declared in pyproject.toml.
- Hosted GET /api/health, /api/classes and /api/metrics returned HTTP 200. Class count is 43; model and metrics accurately report unavailable.
- Hosted confusion-matrix empty state was verified in the preceding deployment with the same API implementation.
- Browser verified final frontend rendering, API connection, privacy wording, image upload and actual missing-model response.
- Existing Vercel access protection retained. No GTSRB model, TensorFlow runtime or experimental results bundled in this hosted build.
- Full trained inference remains unverified and requires the model and a TensorFlow-capable inference runtime.


## Minimal three-page revision

- The current interface contains exactly three navigation pages: Recognition, Model & Training, Dataset. Extra pages, quiz, gallery, modal illustrations and browser history were removed.
- Local TypeScript/Vite build passed. Browser checked all three pages, setup links, training command copy feedback, actual upload/missing-model response, and remove image.
- At 390 px, document width equals viewport width; mobile navigation to Model & Training works.
- Production deployment dpl_FMZxvxZmMnao87r8zB1vX9H7mGNb reached READY; Vercel rebuild passed. Live navigation contains only the three intended pages; training/dataset empty states were checked. Hosted health returned 200.
- The Python ML and API implementations are unchanged; the prior 13 backend tests were not rerun for this frontend-only revision.
- No GTSRB-trained model or measured recognition accuracy has been added.


## Screenshot-requested UI cleanup

- Removed header prototype/connection badges, page project labels, sidebar Micro Project badge, both dataset preparation/setup cards, and the footer.
- TypeScript/Vite build passed. Production deployment dpl_B1DzoqiXLZFK9WAVcB1DkpYPYjvt reached READY.
- Browser verified the unwanted text/cards are absent and there is no footer element. All three navigation pages remain.


## Training cards removal

Removed Train the model and Training status cards, their copy control and unused status request/state. TypeScript/Vite build passed. Deployment dpl_3caQ7BvPNcDvvyLHsv3Knwyynmn4 reached READY. Browser confirmed both cards are absent and the model explanation/results remain.
