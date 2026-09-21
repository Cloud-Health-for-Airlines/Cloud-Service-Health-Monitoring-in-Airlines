# presentation/

Presentation materials and interactive slide deck for **BACCP** (*Boundary-Aware Cross-Generation Cascade Predictor*).

## Presentation Deliverables

- [`slide-deck.md`](./slide-deck.md) — Canonical 15-slide technical presentation deck formatted in [Marp](https://marp.app/)-compatible markdown.
- [`presentation.html`](./presentation.html) — Standalone, interactive browser-based slide runner with zero dependencies, slide counter, and keyboard controls.

---

## 15-Slide Presentation Structure

1. **Title**: BACCP — Boundary-Aware Cross-Generation Cascade Predictor
2. **Problem Statement**: Generational chasm, cross-boundary cascades, reactive APM limits, mainframe black hole
3. **Motivation**: Real-world airline meltdowns (Southwest 2022, Delta 2024, United 2026), need for lead time
4. **Research Gap**: Evolution from traditional monitoring to cloud graphs to BACCP
5. **Objectives**: The 5 core technical project milestones
6. **System Architecture**: Canonical end-to-end execution pipeline with status badges
7. **Legacy ↔ Cloud Boundary**: Boundary gateway architecture and zero-instrumentation eBPF socket tracing
8. **Dependency Graph**: Generation-typed nodes (`legacy`, `boundary-gateway`, `cloud-native`), edge relations, and centrality
9. **AI/ML Prediction Pipeline**: Telemetry $\to$ Graph $\to$ RGCN $\to$ Cascade Probability $\to$ Lead Time $\to$ Conformal Calibration $\to$ Explanation
10. **AWS Cloud Integration Layer**: Roles of CloudWatch, X-Ray, SageMaker, Lambda, and SNS in local vs. AWS mode
11. **Monitoring Dashboard**: Actual SRE console UI layout, component captures, SVG graph, and drift gauge
12. **Predictive Alert & Automated Mitigation Flow**: Injected chaos $\to$ drift surge $\to$ prediction $\to$ Lambda circuit breaker $\to$ SNS alert
13. **Current Implementation Status**: Honest evidence-backed audit across all 16 project components
14. **Experimental Evaluation Plan**: Chaos engineering profiles (`network-delay`, `connection-drop`, `batch-job-stall`) and target benchmarks
15. **Future Work & Roadmap**: Phase-II model training, Constrained MDP RL policy, enterprise CO-RE eBPF, and multi-node EKS

---

## How to Present

### Option A: Interactive Browser Runner (Zero Dependencies)
Open directly in any modern browser:
```bash
open presentation/presentation.html
```
- **Navigation**: Use **Left / Right Arrow Keys**, **Spacebar**, or **PageUp / PageDown**.
- **Slide Counter**: Displays current slide position and total count (1 of 15).

### Option B: Marp Presentation & PDF Export
Using the [Marp CLI](https://marp.app/):
```bash
# Preview in browser with live-reload:
npx @marp-team/marp-cli presentation/slide-deck.md --preview

# Export to standalone presentation HTML:
npx @marp-team/marp-cli presentation/slide-deck.md -o presentation/marp-slides.html

# Export to publication-ready PDF slides:
npx @marp-team/marp-cli presentation/slide-deck.md --pdf -o presentation/baccp-presentation.pdf
```

---

## How to Update Screenshots & Results Later

### 1. Capturing and Embedding Frontend Dashboard Screenshots (Slide 11)
1. **Start the BACCP Backend Server**:
   ```bash
   python3 backend/api/app.py
   ```
2. **Launch the React Dashboard**:
   ```bash
   cd frontend
   npm run dev
   # Dashboard opens at http://localhost:3000
   # Or open standalone runner: open frontend/dist_preview/index.html
   ```
3. **Simulate a Fault**:
   Click on the **Chaos Playground** buttons (`Simulate Delay (High)` or `Simulate Drop (High)`).
   Observe the boundary sync drift gauge jump to $> 70\%$, the alert badge turn red, and the countdown timer activate.
4. **Capture Screenshot**:
   - On macOS: Press `Cmd + Shift + 4` and select the dashboard window.
   - Save the image to `presentation/assets/dashboard-active-cascade.png`.
5. **Embed in Slide Deck**:
   In `presentation/slide-deck.md` (Slide 11), update the content block:
   ```markdown
   ![w:900px](assets/dashboard-active-cascade.png)
   ```

### 2. Updating Evaluation Benchmarks (Slide 14)
When teammate Varad completes Phase-II PyTorch model training in `ai-models/` or live benchmark runs:
1. Record experimental metrics from `results/` or benchmark logs.
2. Update the target table in `presentation/slide-deck.md` (Slide 14) and `presentation/presentation.html` (Slide 14) under the `Evaluation Plan` section.
3. Re-run test suite to confirm zero regressions:
   ```bash
   python3 -m unittest backend/tests/test_backend.py
   ```
