CAFA5_PROMPT_TEMPLATE_WITH_INTERPRO = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence and organism information, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms across three aspects: molecular functions (MF), biological processes (BP), and cellular components (CC). First predict InterPro terms for the protein as a list of entries using the tokens <|INTERPRO_SUMMARY_START|>/<|INTERPRO_SUMMARY_END|>. Each entry should contain InterPro entry ID, name, type, and span of protein sequence it is assigned to. Then for each GO aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth within each aspect and present them using the designated tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, and <|CC_START|>/<|CC_END|> for cellular components. After presenting all GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens, followed by a protein function description using <|FUNCTION_SUMMARY_START|>/<|FUNCTION_SUMMARY_END|> tokens.",
    "user_prompt": "Predict InterPro terms and {go_aspects} GO terms for the given protein from organism {organism}.",
}

CAFA5_PROMPT_TEMPLATE_GO_ONLY = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence and organism information, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms across three aspects: molecular functions (MF), biological processes (BP), and cellular components (CC). For each aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth within each aspect and present them using the designated tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, and <|CC_START|>/<|CC_END|> for cellular components. After presenting all GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens, followed by a protein function description using <|FUNCTION_SUMMARY_START|>/<|FUNCTION_SUMMARY_END|> tokens.",
    "user_prompt": "Predict {go_aspects} GO terms for the given protein from organism {organism}.",
}

# Templates without protein function description (for proteins without function data)
CAFA5_PROMPT_TEMPLATE_WITH_INTERPRO_NO_FUNC = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence and organism information, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms across three aspects: molecular functions (MF), biological processes (BP), and cellular components (CC). First predict InterPro terms for the protein as a list of entries using the tokens <|INTERPRO_SUMMARY_START|>/<|INTERPRO_SUMMARY_END|>. Each entry should contain InterPro entry ID, name, type, and span of protein sequence it is assigned to. Then for each GO aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth within each aspect and present them using the designated tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, and <|CC_START|>/<|CC_END|> for cellular components. After presenting all GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens.",
    "user_prompt": "Predict InterPro terms and {go_aspects} GO terms for the given protein from organism {organism}.",
}

CAFA5_PROMPT_TEMPLATE_GO_ONLY_NO_FUNC = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence and organism information, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms across three aspects: molecular functions (MF), biological processes (BP), and cellular components (CC). For each aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth within each aspect and present them using the designated tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, and <|CC_START|>/<|CC_END|> for cellular components. After presenting all GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens.",
    "user_prompt": "Predict {go_aspects} GO terms for the given protein from organism {organism}.",
}


# New templates for single GO aspect prediction
CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_WITH_INTERPRO = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence and organism information, predict Gene Ontology (GO) terms for a specific aspect. Provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens, followed by a protein function description using <|FUNCTION_SUMMARY_START|>/<|FUNCTION_SUMMARY_END|> tokens.",
    "user_prompt": "Predict InterPro terms and {go_aspects} GO terms for the given protein from organism {organism}.",
}

CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_GO_ONLY = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence and organism information, predict Gene Ontology (GO) terms for a specific aspect. Provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens, followed by a protein function description using <|FUNCTION_SUMMARY_START|>/<|FUNCTION_SUMMARY_END|> tokens.",
    "user_prompt": "Predict {go_aspects} GO terms for the given protein from organism {organism}.",
}

# Templates for single aspect without protein function description
CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_WITH_INTERPRO_NO_FUNC = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence and organism information, predict Gene Ontology (GO) terms for a specific aspect. Provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens, followed by a protein function description using <|FUNCTION_SUMMARY_START|>/<|FUNCTION_SUMMARY_END|> tokens.",
    "user_prompt": "Predict InterPro terms and {go_aspects} GO terms for the given protein from organism {organism}.",
}

CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_GO_ONLY_NO_FUNC = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence and organism information, predict Gene Ontology (GO) terms for a specific aspect. Provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens, followed by a protein function description using <|FUNCTION_SUMMARY_START|>/<|FUNCTION_SUMMARY_END|> tokens.",
    "user_prompt": "Predict {go_aspects} GO terms for the given protein from organism {organism}.",
}



# New templates with InterPro data included in user prompt
CAFA5_PROMPT_TEMPLATE_INTERPRO_IN_PROMPT = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, and InterPro domain annotations, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms across three aspects: molecular functions (MF), biological processes (BP), and cellular components (CC). Use the provided InterPro information to guide your predictions. For each GO aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth within each aspect and present them using the designated tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, and <|CC_START|>/<|CC_END|> for cellular components. After presenting all GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens, followed by a protein function description using <|FUNCTION_SUMMARY_START|>/<|FUNCTION_SUMMARY_END|> tokens.",
    "user_prompt": "Given the protein above from organism {organism} with the following InterPro annotations:\n{interpro_data}\n\nPredict {go_aspects} GO terms for this protein.",
}

CAFA5_PROMPT_TEMPLATE_INTERPRO_IN_PROMPT_NO_FUNC = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, and InterPro domain annotations, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms across three aspects: molecular functions (MF), biological processes (BP), and cellular components (CC). Use the provided InterPro information to guide your predictions. For each GO aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth within each aspect and present them using the designated tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, and <|CC_START|>/<|CC_END|> for cellular components. After presenting all GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens.",
    "user_prompt": "Given the protein above from organism {organism} with the following InterPro annotations:\n{interpro_data}\n\nPredict {go_aspects} GO terms for this protein.",
}

# Single aspect templates with InterPro in prompt
CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_IN_PROMPT = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, and InterPro domain annotations, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms for a specific aspect. Use the provided InterPro information to guide your predictions. For the specified GO aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth and present them using the appropriate tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, or <|CC_START|>/<|CC_END|> for cellular components. After presenting the GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens, followed by a protein function description using <|FUNCTION_SUMMARY_START|>/<|FUNCTION_SUMMARY_END|> tokens.",
    "user_prompt": "Given the protein above from organism {organism} with the following InterPro annotations:\n{interpro_data}\n\nPredict {go_aspects} GO terms for this protein.",
}

CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_IN_PROMPT_NO_FUNC = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, and InterPro domain annotations, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms for a specific aspect. Use the provided InterPro information to guide your predictions. For the specified GO aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth and present them using the appropriate tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, or <|CC_START|>/<|CC_END|> for cellular components. After presenting the GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens.",
    "user_prompt": "Given the protein above from organism {organism} with the following InterPro annotations:\n{interpro_data}\n\nPredict {go_aspects} GO terms for this protein.",
}

# Templates with PPI data only in user prompt
CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_PPI_IN_PROMPT = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, and protein-protein interaction (PPI) data, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms for a specific aspect. For the specified GO aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth and present them using the appropriate tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, or <|CC_START|>/<|CC_END|> for cellular components. After presenting the GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens, followed by a protein function description using <|FUNCTION_SUMMARY_START|>/<|FUNCTION_SUMMARY_END|> tokens.",
    "user_prompt": "Given the protein above from organism {organism} with the following protein-protein interaction partners:\n{ppi_data}\n\nPredict {go_aspects} GO terms for this protein.",
}

CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_PPI_IN_PROMPT_NO_FUNC = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, and protein-protein interaction (PPI) data, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms for a specific aspect. For the specified GO aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth and present them using the appropriate tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, or <|CC_START|>/<|CC_END|> for cellular components. After presenting the GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens.",
    "user_prompt": "Given the protein above from organism {organism} with the following protein-protein interaction partners:\n{ppi_data}\n\nPredict {go_aspects} GO terms for this protein.",
}

# Templates with both InterPro and PPI data in user prompt
CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_PPI_IN_PROMPT = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, InterPro domain annotations, and protein-protein interaction (PPI) data, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms for a specific aspect. Use the provided InterPro and PPI information to guide your predictions. For the specified GO aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth and present them using the appropriate tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, or <|CC_START|>/<|CC_END|> for cellular components. After presenting the GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens, followed by a protein function description using <|FUNCTION_SUMMARY_START|>/<|FUNCTION_SUMMARY_END|> tokens.",
    "user_prompt": "Given the protein above from organism {organism} with the following InterPro annotations:\n{interpro_data}\n\nAnd the following protein-protein interaction partners:\n{ppi_data}\n\nPredict {go_aspects} GO terms for this protein.",
}

CAFA5_PROMPT_TEMPLATE_SINGLE_ASPECT_INTERPRO_PPI_IN_PROMPT_NO_FUNC = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, InterPro domain annotations, and protein-protein interaction (PPI) data, analyze the protein and predict its functional annotations using Gene Ontology (GO) terms for a specific aspect. Use the provided InterPro and PPI information to guide your predictions. For the specified GO aspect, systematically traverse the GO hierarchy from general root terms to specific leaf terms, predicting relevant child terms at progressively deeper levels. Organize your predictions by depth and present them using the appropriate tokens: <|MF_START|>/<|MF_END|> for molecular functions, <|BP_START|>/<|BP_END|> for biological processes, or <|CC_START|>/<|CC_END|> for cellular components. After presenting the GO predictions, provide a comprehensive list using <|GO_SUMMARY_START|>/<|GO_SUMMARY_END|> tokens.",
    "user_prompt": "Given the protein above from organism {organism} with the following InterPro annotations:\n{interpro_data}\n\nAnd the following protein-protein interaction partners:\n{ppi_data}\n\nPredict {go_aspects} GO terms for this protein.",
}

# Reasoning template without special tokens for comprehensive analysis
CAFA5_REASONING_TEMPLATE = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence and organism information, step-by-step reason about the InterPro terms, Gene Ontology (GO) terms regarding molecular function, biological process, and cellular component, protein-protein interactions (PPI), and overall function. Provide a summary of your findings in your final answer.",
    "user_prompt": "Given the protein above from organism {organism}, reason about the function of the protein.",
}

# Reasoning template for SwissProt with dynamic content based on available data
CAFA5_REASONING_TEMPLATE_SWISSPROT = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence and organism information, step-by-step reason about the InterPro terms{go_terms_text}{ppi_text}, and overall function. Provide a summary of your findings in your final answer.",
    "user_prompt": "Given the protein above from organism {organism}, reason about the function of the protein.",
}

# Reasoning template with InterPro and/or GO speculations
CAFA5_REASONING_TEMPLATE_WITH_CONTEXT = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, and additional context (InterPro domain annotations and/or initial GO term speculations), step-by-step reason about the InterPro terms, Gene Ontology (GO) terms regarding molecular function, biological process, and cellular component, protein-protein interactions (PPI), and overall function. Use the provided information as a starting point and improve upon it with deeper analysis. Provide a summary of your findings in your final answer.",
    "user_prompt": "Given the protein above from organism {organism} with the following InterPro annotations:\n{interpro_data}\n\nAnd the following initial GO term speculations:\n{go_speculations}\n\nReason about the function of the protein.",
}

# Reasoning template with InterPro and/or GO speculations with PPI
CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, and additional context (InterPro domain annotations and/or initial GO term speculations), step-by-step reason about the InterPro terms, Gene Ontology (GO) terms regarding molecular function, biological process, and cellular component, protein-protein interactions (PPI), and overall function. Use the provided information as a starting point and improve upon it with deeper analysis. Provide a summary of your findings in your final answer.",
    "user_prompt": "Given the protein above from organism {organism} with the following InterPro annotations:\n{interpro_data}\n\nAnd the following protein-protein interaction partners:\n{ppi_data}\n\nAnd the following initial GO term speculations:\n{go_speculations}\n\nReason about the function of the protein{go_aspects_suffix}",
}

# Reasoning template with InterPro and/or GO speculations with PPI
CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI_UNIPROT = {
    "system_prompt": "You are a scientific assistant specialized in protein function prediction. Given a protein sequence, organism information, and additional context (InterPro domain annotations and/or initial GO term speculations), step-by-step reason about the InterPro terms, Gene Ontology (GO) terms regarding molecular function, biological process, and cellular component, protein-protein interactions (PPI), and overall function. Use the provided information as a starting point and improve upon it with deeper analysis. Provide a summary of your findings in your final answer.",
    "user_prompt": "Given the protein above from organism {organism} with the following InterPro annotations:\n{interpro_data}\n\nAnd the following protein-protein interaction partners:\n{ppi_data}\n\nAnd the following initial GO term speculations:\n{go_speculations}\n\nReason about the function of the protein{go_aspects_suffix}{uniprot_summary}",
}

# Paper-compact reasoning template for RL continuation tuning.
# The text context is intentionally restricted to the paper-style slots;
# protein residue and GO graph embeddings are provided separately.
CAFA5_REASONING_TEMPLATE_PAPER_NATIVE = {
    "system_prompt": (
        "You are a scientific assistant specialized in protein function prediction. "
        "The protein residue embeddings and 200 GO graph embeddings are already provided "
        "as multimodal context. The user prompt is assembled by concatenating the target "
        "organism, InterPro domain annotations (identifiers, names, and residue ranges), "
        "greedy-decoded GO-GPT predictions, and protein-protein interaction partners when "
        "available. Produce exactly two sections in order: <|REASONING|> ... <|/REASONING|> "
        "followed by <|FINAL_ANSWER|> ... <|/FINAL_ANSWER|>. Use only the provided evidence, "
        "revise weak GO-GPT hypotheses when needed, and do not emit templates, placeholders, "
        "tool-call text, or meta-instructions."
    ),
    "user_prompt": (
        "Organism: {organism}\n\n"
        "InterPro annotations (identifiers, names, residue ranges):\n{interpro_data}\n\n"
        "Greedy-decoded GO-GPT predictions:\n"
        "Molecular Function (MF): {go_mf_speculations}\n"
        "Biological Process (BP): {go_bp_speculations}\n"
        "Cellular Component (CC): {go_cc_speculations}\n\n"
        "Protein-protein interaction partners:\n{ppi_data}\n\n"
        "Produce a structured reasoning trace followed by a final answer. "
        "In <|FINAL_ANSWER|>, list real GO IDs one per line before any concise function summary "
        "or hypothesized interaction partners."
    ),
}

# Paper-compact reasoning template for structured ablations.
# This is intentionally stricter than the paper-native continuation contract.
CAFA5_REASONING_TEMPLATE_PAPER_COMPACT = {
    "system_prompt": (
        "You are a scientific assistant specialized in protein function prediction. "
        "The protein residues and GO graph are already provided as multimodal context. "
        "Given organism information, InterPro annotations, protein-protein interaction "
        "(PPI) partners, and initial GO term hypotheses, reason briefly and then provide "
        "the final answer as early as possible. Keep any free-form reasoning to at most "
        "two short sentences. Do not write a long narrative explanation. "
        "Output exactly one structured GO summary block enclosed by <|GO_SUMMARY_START|> "
        "and <|GO_SUMMARY_END|>, then stop immediately. Only include GO terms for the "
        "requested focus aspect unless all three aspects are explicitly requested. Each "
        "line in the GO summary must use the format `ASPECT: GO:XXXXXXX (term name); "
        "GO:YYYYYYY (term name)` or `ASPECT: None`. The X/Y GO IDs are placeholders "
        "only and must be replaced with real supported GO IDs. Do not add a UniProt-style "
        "prose summary."
    ),
    "user_prompt": (
        "Organism: {organism}\n"
        "InterPro annotations:\n{interpro_data}\n\n"
        "PPI partners:\n{ppi_data}\n\n"
        "Initial GO term speculations:\n"
        "Molecular Function (MF): {go_mf_speculations}\n"
        "Biological Process (BP): {go_bp_speculations}\n"
        "Cellular Component (CC): {go_cc_speculations}\n\n"
        "Focus aspect: {focus_aspect} ({focus_aspect_code})\n\n"
        "Required final answer format:\n"
        "{response_format_example}\n\n"
        "Important: the GO:XXXXXXX and GO:YYYYYYY strings in the example are placeholders only. "
        "Do not copy them into the final answer. Replace them with real GO IDs from the evidence, "
        "or write `ASPECT: None` when no supported GO term is justified.\n\n"
        "Reason briefly from the evidence above, then produce the final answer block and stop. "
        "Begin the final answer with <|GO_SUMMARY_START|> on its own line."
    ),
}

# Helper prompts
INTERPRO_IN_GENERATION = "Let me list the InterPro terms for this protein."
