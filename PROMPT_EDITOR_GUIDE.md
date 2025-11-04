# Model Analysis Prompt Editor Guide

## Overview

The Research Paper Assistant now features **3 customizable analysis prompts** that you can edit directly on the main page!

## What's New?

### 📝 Three Separate Prompts

You can now customize **all three** analysis types:

1. **📊 General Analysis** - Overall paper evaluation
2. **🔬 Methodology Focus** - Deep dive into research methods
3. **📈 Results Focus** - Analysis of findings and data

### 🎯 Selecting Prompt Type

**You can choose which prompt to use!** Go to the sidebar under **"🎯 Analysis Focus"** and select:

- **📊 General Analysis** (default) - Overall paper evaluation
- **🔬 Methodology Focus** - Deep dive into research methods
- **📈 Results Focus** - Analysis of findings and data

The selected prompt type determines which prompt all models will use for their analysis.

## Location

The prompt editor appears on the **Main Analysis Tab**, right after the text input area:

```
📊 Analysis Tab
├── [Upload PDF]
├── [Paste text]
├── ─────────────────
├── 📝 Model Analysis Prompts  ← HERE!
│   ├── Tab: 📊 General Analysis
│   ├── Tab: 🔬 Methodology Focus
│   ├── Tab: 📈 Results Focus
├── ─────────────────
└── [Analyze Paper] button
```

## Features

### ✅ Each Prompt Tab Includes:
- **Large text editor** (220px height) - Easy to read and edit
- **🔄 Reset button** - Restore default prompt with one click
- **Helpful caption** - Explains what each prompt does
- **Session persistence** - Your edits are saved while using the app

### ✅ During Analysis:
- **Expandable section** shows which prompts were used
- All 3 prompts are displayed for transparency
- You can verify the exact prompt that was sent to the models

## Default Prompts

### 📊 General Analysis (Currently Used)
```
Analyze this research paper systematically:

1. **Main Topic & Objective**: What research question does this address? (2-3 sentences)
2. **Key Methodology**: What approach/methods were used? Include specific techniques. (3-4 sentences)
3. **Major Findings**: What were the main results? Include metrics or data if present. (3-4 sentences)
4. **Strengths**: What does the paper do well? (2-3 specific points with evidence)
5. **Areas for Improvement**: What could be enhanced? (2-3 specific points)

Cite specific evidence from the paper.
```

### 🔬 Methodology Focus
```
Analyze the methodology in detail:

1. **Approach Overview**: Summarize main methods and experimental design
2. **Rigor & Validity**: Evaluate methodological soundness
3. **Data & Analysis**: Assess data collection and analytical techniques
4. **Reproducibility**: How clearly are methods described?
5. **Limitations**: Identify gaps or weaknesses

Provide evidence-based critique with examples.
```

### 📈 Results Focus
```
Analyze the results comprehensively:

1. **Key Findings**: Summarize main results with specific metrics
2. **Presentation Quality**: Evaluate tables, figures, data visualization
3. **Statistical Rigor**: Assess validity of statistical methods
4. **Results vs Claims**: Do results support conclusions?
5. **Completeness**: What additional analysis would help?

Focus on specific data points and quantitative measures.
```

## How to Use

### Step 1: Select Prompt Type (Sidebar)
1. Look in the **sidebar** under **"🎯 Analysis Focus"**
2. Check **"Model Comparison"**
3. Select which prompt type you want:
   - 📊 General Analysis (default)
   - 🔬 Methodology Focus
   - 📈 Results Focus

### Step 2: Edit Prompts (Main Page)
1. Scroll down to the **"📝 Model Analysis Prompts"** section on the main page
2. You'll see an info box showing which prompt is currently selected
3. Click on any of the 3 tabs to edit the corresponding prompt
4. The tab for your selected prompt type is the one that will be used
5. Click **🔄 Reset** to restore defaults if needed

### Step 3: Analyze Paper
1. Upload a PDF or paste text
2. Click **"Analyze Paper"** button
3. The analysis will use your selected prompt type with any customizations

### Step 4: Verify (Optional)
1. After analysis completes, expand **"📝 Using Prompt: [Type]"**
2. See exactly which prompt was used and verify it matches your selection
3. Optionally expand "Show all 3 prompts" to see what all prompts were

## Customization Tips

### 💡 Make Prompts More Specific
Instead of:
```
What research question does this address?
```

Try:
```
What specific research gap does this paper address in the field of machine learning?
How does it relate to recent work on transformer architectures?
```

### 💡 Add Domain-Specific Questions
For a medical paper:
```
1. **Clinical Relevance**: How applicable are the findings to clinical practice?
2. **Patient Safety**: Are there any safety concerns or contraindications?
3. **Statistical Power**: Was the sample size adequate?
```

### 💡 Focus on Your Interests
If you only care about methodology:
```
Evaluate this paper's methodology:

1. Was the experimental design appropriate?
2. Are there any confounding variables?
3. Can the study be replicated based on the description?
4. What are the main methodological limitations?
```

## Technical Details

### Backend Implementation
- **File**: `model_comparison_analyzer.py`
- **Function**: `analyze_paper_single_model()`
- **Parameter**: `custom_prompts` (dict with keys: "general", "methodology", "results")
- **Fallback**: If custom prompts are empty or None, defaults are used

### Frontend Implementation
- **File**: `research_paper_assistant_hybrid_citation.py`
- **Session State Keys**:
  - `st.session_state.prompt_general`
  - `st.session_state.prompt_methodology`
  - `st.session_state.prompt_results`
- **UI Components**: Tabs with text_area widgets and reset buttons

### Data Flow
```
UI (3 text areas)
    ↓
Session State (3 prompts)
    ↓
Settings dict {"prompt_general", "prompt_methodology", "prompt_results"}
    ↓
Model Analyzer (custom_prompts parameter)
    ↓
AI Models (receive appropriate prompt)
```

## Future Enhancements

Potential future features:
- [ ] Dropdown to select which prompt type to use (general/methodology/results)
- [ ] Save/load custom prompt templates
- [ ] Prompt library with pre-made templates for different fields
- [ ] Run all 3 analyses (general + methodology + results) simultaneously
- [ ] Compare outputs from different prompt types

## Troubleshooting

### Prompt Not Being Used
- **Check:** Verify you clicked "Analyze Paper" after editing
- **Solution:** Re-run the analysis after making changes

### Reset Button Not Working
- **Check:** Make sure you clicked the correct reset button for the active tab
- **Solution:** Each tab has its own reset button

### Changes Lost After Refresh
- **Expected:** Session state is cleared on page refresh
- **Solution:** Copy your custom prompts to a text file before refreshing

## Examples

### Example 1: Focus on Reproducibility
```
Evaluate this paper's reproducibility:

1. **Code Availability**: Is code/data publicly available?
2. **Implementation Details**: Are hyperparameters and settings documented?
3. **Environment**: Is the computational environment specified?
4. **Replication Instructions**: Could another researcher replicate this?
5. **Missing Details**: What information is needed for reproduction?
```

### Example 2: Critical Review
```
Provide a critical analysis:

1. **Novelty**: What is genuinely new vs. incremental?
2. **Evidence Quality**: How strong is the supporting evidence?
3. **Limitations**: What are the main weaknesses?
4. **Alternative Explanations**: Are there other interpretations?
5. **Follow-up Research**: What questions remain unanswered?
```

### Example 3: Quick Summary
```
Provide a concise summary:

1. What problem does this solve? (1 sentence)
2. What's the approach? (1 sentence)
3. What are the results? (1 sentence)
4. Why does it matter? (1 sentence)
```

---

**Happy analyzing!** 🚀

For more information, see `CHANGES_SUMMARY.md`.
