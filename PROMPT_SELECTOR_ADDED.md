# Prompt Type Selector - Now Live! 🎉

## What's New?

You can now **choose which prompt type to use** for model analysis!

## Where to Find It?

### Sidebar → "🎯 Analysis Focus"

When you enable **"Model Comparison"**, you'll see:

```
Model Analysis Type:
Choose which prompt to use:
  ○ 📊 General Analysis
  ○ 🔬 Methodology Focus
  ○ 📈 Results Focus
```

## How It Works

### 1️⃣ **Select in Sidebar**
Choose your desired analysis type using the radio buttons.

### 2️⃣ **Edit on Main Page**
The main page shows which prompt is currently selected:
```
✓ Currently selected: 📊 General Analysis (change in sidebar)
```

### 3️⃣ **Analyze Paper**
Click "Analyze Paper" and **all models** will use the selected prompt type!

### 4️⃣ **Verify Results**
After analysis, the expandable section shows:
```
📝 Using Prompt: 📊 General Analysis
```

## Example Workflow

### Scenario: You want to deeply analyze the methodology

**Step 1:** Go to sidebar → Select **🔬 Methodology Focus**

**Step 2:** (Optional) Edit the methodology prompt on main page:
```
Analyze the methodology in detail:

1. Approach Overview: Summarize methods and design
2. Rigor & Validity: Evaluate soundness
3. Data & Analysis: Assess techniques
4. Reproducibility: Check clarity
5. Limitations: Identify gaps
```

**Step 3:** Click **"Analyze Paper"**

**Step 4:** All 4 models (Deepseek 1.5B, 8B, Mistral, LLaMA3) will analyze the paper focusing on methodology!

## Visual Guide

```
┌─────────────────────────────────────┐
│         SIDEBAR                     │
├─────────────────────────────────────┤
│ 🎯 Analysis Focus                   │
│   ☑ Citation Analysis              │
│   ☑ Model Comparison               │
│                                     │
│ Model Analysis Type:                │
│   ● 📊 General Analysis      ← SELECT HERE
│   ○ 🔬 Methodology Focus            │
│   ○ 📈 Results Focus                │
└─────────────────────────────────────┘

         ⬇️

┌─────────────────────────────────────┐
│       MAIN PAGE                     │
├─────────────────────────────────────┤
│ 📝 Model Analysis Prompts           │
│                                     │
│ ✓ Currently selected:               │
│   📊 General Analysis               │
│   (change in sidebar)               │
│                                     │
│ [Tab: 📊 General Analysis]         │ ← EDIT HERE
│ [Tab: 🔬 Methodology Focus]        │
│ [Tab: 📈 Results Focus]            │
│                                     │
│ [Analyze Paper]                    │
└─────────────────────────────────────┘

         ⬇️

┌─────────────────────────────────────┐
│         RESULTS                     │
├─────────────────────────────────────┤
│ 🤖 Enhanced Model Analysis          │
│                                     │
│ ▼ 📝 Using Prompt:                 │
│    📊 General Analysis              │ ← VERIFY HERE
│                                     │
│   Active prompt type:               │
│   📊 General Analysis               │
│                                     │
│   [General Analysis Prompt shown]   │
│                                     │
│   ▼ Show all 3 prompts             │
└─────────────────────────────────────┘
```

## All 3 Prompt Types Explained

### 📊 General Analysis (Default)
**Best for:** Overall paper review, first-time reading
**Focuses on:** Main topic, methodology overview, key findings, strengths, areas for improvement

### 🔬 Methodology Focus
**Best for:** Evaluating experimental design and research rigor
**Focuses on:** Methods, validity, data analysis, reproducibility, limitations

### 📈 Results Focus
**Best for:** Deep dive into findings and statistical analysis
**Focuses on:** Key findings with metrics, presentation quality, statistical rigor, claims vs results

## Switch Anytime!

You can:
- ✅ Switch prompt types between analyses
- ✅ Edit all 3 prompts independently
- ✅ Reset individual prompts to defaults
- ✅ See which prompt was used in results

## Technical Implementation

### Files Modified:
- `research_paper_assistant_hybrid_citation.py` (lines 264-280, 776-804, 829)
  - Added radio button selector in sidebar
  - Display selected type on main page
  - Pass selected type to model analyzer
  - Show active prompt in results

### Key Changes:
```python
# Sidebar selector
analysis_type = st.radio(
    "Choose which prompt to use:",
    options=["general", "methodology", "results"],
    ...
)

# Pass to analyzer
result = self.model_analyzer.analyze_paper_single_model(
    text,
    model_name,
    model,
    analysis_type=settings.get("analysis_type", "general"),  # ✨ Dynamic!
    ...
)
```

## Benefits

✅ **Flexibility** - Choose the right analysis for your needs
✅ **Transparency** - Always know which prompt is being used
✅ **Customization** - Edit any prompt independently
✅ **Consistency** - All models use the same selected prompt
✅ **Easy switching** - Change types without editing prompts

## Quick Reference

| Want to... | Use Prompt Type | Why |
|-----------|----------------|-----|
| Get a complete overview | 📊 General | Covers everything |
| Evaluate research methods | 🔬 Methodology | Focuses on approach |
| Analyze findings | 📈 Results | Deep dive into data |
| Check reproducibility | 🔬 Methodology | Assesses clarity |
| Compare to other work | 📊 General | Includes strengths/weaknesses |

---

**Try it now:**
```bash
streamlit run research_paper_assistant_hybrid_citation.py
```

1. Open sidebar
2. Enable "Model Comparison"
3. Select a prompt type
4. Upload a paper
5. Click "Analyze Paper"
6. See the difference! 🚀
