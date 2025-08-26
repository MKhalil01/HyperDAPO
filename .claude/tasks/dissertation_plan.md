# Dissertation Plan: HyperDAPO (ACCELERATED - 7 DAY TIMELINE)
## DAPO-Based Framework for Optimal Execution of Hyperliquid Limit Orders

**Student:** Mohammed Khalil  
**Supervisor:** Arthur Gervais  
**Submission Date:** September 8, 2025  
**Program:** MSc Information Security, UCL

---

## 🚨 ACCELERATED 7-DAY WRITING SCHEDULE

### Day 1 (Monday) - Foundation Sprint
**Morning (4 hours)**
- Write complete Introduction chapter (5 pages)
  - Problem statement, research questions, contributions
  - Use existing THESIS_RESULTS_SUMMARY.md content

**Afternoon (4 hours)**
- Write Methodology sections 3.1-3.2 (5 pages)
  - System architecture (use existing diagrams)
  - Data collection (already documented)

### Day 2 (Tuesday) - Technical Core
**Morning (4 hours)**
- Complete Methodology sections 3.3-3.4 (5 pages)
  - LLM design (from existing code)
  - DAPO training (from IMPLEMENTATION_PLAN.md)

**Afternoon (4 hours)**
- Write Implementation chapter (8 pages)
  - Technical stack, training infrastructure
  - Copy from existing documentation

### Day 3 (Wednesday) - Results & Background
**Morning (4 hours)**
- Write Results chapter sections 5.1-5.2 (6 pages)
  - Use existing metrics from THESIS_RESULTS_SUMMARY.md
  - Generate tables/plots from backtest_results.csv

**Afternoon (4 hours)**
- Write Background chapter sections 2.1-2.2 (5 pages)
  - Adapt from literature review (main.pdf)
  - Focus on essential theory only

### Day 4 (Thursday) - Analysis & Background Completion
**Morning (4 hours)**
- Complete Background sections 2.3-2.4 (5 pages)
- Complete Results sections 5.3-5.4 (4 pages)

**Afternoon (4 hours)**
- Write Discussion chapter (6 pages)
  - Key findings, limitations, implications
  - Use existing evaluation results

### Day 5 (Friday) - Conclusion & Polish
**Morning (4 hours)**
- Write Conclusion chapter (3 pages)
- Write Abstract (1 page)
- Create Table of Contents

**Afternoon (4 hours)**
- Format all figures/tables
- Insert all citations
- First complete draft review

### Day 6 (Saturday) - Refinement
**Morning (4 hours)**
- Address any gaps
- Improve transitions between chapters
- Check technical accuracy

**Afternoon (4 hours)**
- Proofreading pass 1
- Fix LaTeX formatting issues
- Ensure 50-page limit compliance

### Day 7 (Sunday) - Final Polish
**Morning (4 hours)**
- Final proofreading
- Reference checking
- Format compliance check

**Afternoon (4 hours)**
- Generate final PDF
- Backup everything
- Submit

---

## PRIORITY CONTENT STRATEGY

### What to INCLUDE (Essential)
✅ Clear problem statement and contributions  
✅ DAPO methodology explanation  
✅ Implementation overview  
✅ Key results (72% fill rate, -1.00 bps slippage)  
✅ Statistical significance tests  
✅ Comparison with baselines  
✅ Critical limitations discussion  

### What to MINIMIZE (Time-savers)
❌ Extensive literature review (keep to 5-6 key papers)  
❌ Detailed mathematical proofs (reference instead)  
❌ Extended background theory (focus on essentials)  
❌ Multiple ablation studies (1-2 max)  
❌ Lengthy future work (bullet points)  

### What to REUSE (From existing docs)
📄 THESIS_RESULTS_SUMMARY.md → Results chapter  
📄 IMPLEMENTATION_PLAN.md → Methodology  
📄 TECHNICAL_NOTES.md → Implementation details  
📄 main.pdf (lit review) → Background chapter  
📄 Existing code → Technical appendix  

---

## DAILY WRITING TARGETS

| Day | Chapters | Target Pages | Cumulative |
|-----|----------|--------------|------------|
| 1 | Intro + Method (partial) | 10 | 10 |
| 2 | Method (complete) + Implementation | 13 | 23 |
| 3 | Results + Background (partial) | 11 | 34 |
| 4 | Background (complete) + Discussion | 15 | 49 |
| 5 | Conclusion + Abstract + ToC | 5 | 50 (main text) |
| 6-7 | Refinement only | 0 | 50 |

---

## WRITING TIPS FOR SPEED

### Use Templates
```latex
% For each chapter introduction
This chapter presents [topic]. Section X.1 introduces [subtopic1], 
followed by [subtopic2] in Section X.2...

% For transitions
Building on the [previous concept], we now examine...
Having established [X], we turn to [Y]...
```

### Quick Wins
1. **Tables/Figures** count toward page limit - use liberally
2. **Code blocks** in appendix don't count - move details there
3. **Bullet points** for future work save space
4. **Citations** can be minimal - focus on key papers
5. **Equations** take space - use them strategically

### Emergency Shortcuts
- If behind schedule, prioritize: Intro → Results → Methodology → Discussion
- Background can be trimmed most without losing core contribution
- Implementation details can go to appendix
- Use existing diagrams/plots - don't create new ones

---

## CRITICAL PATH ITEMS

### Must Complete First
1. Introduction (sets entire narrative)
2. Results tables/figures (core evidence)
3. Methodology DAPO section (novel contribution)

### Can Be Simplified
1. Literature review depth
2. Mathematical formulations
3. Extended evaluation metrics
4. Detailed implementation code

### Pre-Written Content Available
- Training results → THESIS_RESULTS_SUMMARY.md
- Backtesting metrics → backtest_results.csv, thesis_metrics.json
- Model details → MODEL_RESPONSES_SUMMARY.txt
- Technical implementation → src/ code files

---

## EMERGENCY CONTINGENCY

If falling behind by Day 3:
- Reduce Background to 6 pages (minimum viable)
- Combine Implementation into Methodology
- Simplify Discussion to 4 pages
- Focus on core narrative: problem → solution → results → impact

Remember: **Done is better than perfect** - aim for submission-ready, not perfection!