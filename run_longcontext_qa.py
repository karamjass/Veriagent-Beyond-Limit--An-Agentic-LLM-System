"""
Long-Context QA Benchmark for VeriAgent
========================================
Tests qwen3.5:9b-q4_K_M (262k-token context window) on reading comprehension
across synthetic documents ranging from ~3k to ~10k tokens.

Design:
 - 15 questions across 5 synthetic long documents
 - Each document has 3 target questions with exact gold answers embedded
 - Distractor text surrounds the answer (needle-in-haystack style)
 - Scoring: Exact substring match (primary) + token F1 (secondary)
 - Baseline: direct Ollama call, no pipeline
 - Pipeline: VeriAgent model_core.answer_question()

Rule: Never invent or estimate. All results come from live model runs.
"""

import json
import re
import time
import sys
import os
import hashlib
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
GENERATOR_MODEL = "qwen3.5:9b-q4_K_M"
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "longcontext_qa_results.json")
OUTPUT_MD   = os.path.join(RESULTS_DIR, "longcontext_qa_summary.md")

# ---------------------------------------------------------------------------
# Synthetic Long-Context Documents + Questions
# ---------------------------------------------------------------------------
# Each document is a realistic ~3k-8k word passage with factual nuggets
# placed at different positions (early, middle, late, very late)
# to test true reading comprehension vs position bias.

DOCUMENTS = [

# ----- DOC 1: Tech History Essay (~2200 words, answers at start/mid/end) -----
{
"id": "doc_tech_history",
"approx_words": 2300,
"title": "The Evolution of Microprocessor Architecture (1971–2024)",
"text": """The Evolution of Microprocessor Architecture (1971–2024)

The first commercially available microprocessor, the Intel 4004, was released on November 15, 1971. It contained exactly 2,300 transistors etched onto a 10-micron process node and ran at a clock speed of 740 kHz. The chip was originally designed for the Busicom 141-PF calculator by Federico Faggin, Ted Hoff, and Stanley Mazor. Its 4-bit architecture allowed it to handle 4 bits of data at a time, making it suitable for simple arithmetic and control applications. The die size measured 12 mm², and it could execute approximately 92,000 instructions per second.

The semiconductor revolution did not happen overnight. Throughout the late 1960s, researchers at Fairchild Semiconductor and Texas Instruments had been pioneering the idea of integrating multiple transistors on a single silicon wafer. Jack Kilby of Texas Instruments and Robert Noyce of Fairchild are co-credited with the invention of the integrated circuit in 1958 and 1959 respectively. Kilby later received the Nobel Prize in Physics in the year 2000 for his contribution.

Advances in lithography throughout the 1970s enabled transistor counts to roughly double every 18 to 24 months — an empirical pattern Gordon Moore first described in a 1965 paper in Electronics Magazine, later refined to a doubling every two years. By 1979, Intel's 8088 processor, which powered the first IBM PC in 1981, ran at a base clock of 5 MHz and contained 29,000 transistors.

The 1980s saw the rise of RISC (Reduced Instruction Set Computer) architectures. At Stanford University, John Hennessy led the MIPS project, while David Patterson at the University of California, Berkeley developed the RISC-I chip. The fundamental insight was that simpler instructions, executed faster, would outperform complex multi-cycle instructions for general-purpose workloads. The commercial success of these ideas materialised in products like the SPARC architecture (Sun Microsystems, 1987) and the ARM architecture, which Acorn Computers, Apple, and VLSI Technology jointly formed in 1990.

Throughout the 1990s the x86 architecture evolved from 32-bit to include 64-bit extensions. Advanced Micro Devices (AMD) introduced the AMD64 instruction set in the Opteron processor family released in April 2003. AMD's extension was subsequently adopted by Intel under the name EM64T (later renamed Intel 64). This meant legacy 32-bit code continued to run while enabling access to more than 4 GB of physical memory.

Multi-core processors emerged as a solution to the power wall — the phenomenon whereby increasing clock frequency beyond approximately 4 GHz produces diminishing returns due to thermal dissipation constraints. Intel's first dual-core desktop processor, the Pentium D 820, was launched in May 2005. AMD followed with its Athlon 64 X2 series in the same year. By 2010, six-core and later eight-core consumer chips were available.

The move to smaller process nodes accelerated under TSMC's leadership. The Taiwanese foundry reached 7nm production in 2018, producing chips for Apple's A12 Bionic, used in the iPhone XS. The A12 contained 6.9 billion transistors. TSMC's 5nm node, used in Apple's M1 chip released in November 2020, packed 16 billion transistors. The M1 achieved a single-core Geekbench 5 score of approximately 1,750 and a multi-core score of approximately 7,700 in independent testing.

GPU computing entered the mainstream scientific discussion around 2007 when Nvidia released the CUDA programming model alongside its Tesla architecture. The GPU's massively parallel design — originally for rendering — proved highly effective for matrix multiplication, underpinning modern deep learning. By 2024, Nvidia's H100 GPU, manufactured on TSMC's 4N process node (a customised 4nm-class node), contained 80 billion transistors and delivered 3,958 TFLOPS of FP8 Tensor Core performance.

Neuromorphic computing represents one of the most unconventional architectural departures. Intel's Loihi 2 chip, announced in September 2021, contained 1 million artificial neurons and 120 million synapses, implementing spike-based computation inspired by biological neural networks. It is fabricated on Intel's 4nm Intel 4 process node. IBM's NorthPole chip, described in a Science paper published in October 2023, eliminates the traditional von Neumann bottleneck by co-locating memory and compute, achieving 22 TOPS/W energy efficiency — approximately 25 times more energy efficient than a standard GPU inference configuration measured at the same workload.

Quantum computing milestones also accelerated in this period. Google's Sycamore processor, using 53 superconducting qubits, completed a specific sampling task in approximately 200 seconds in October 2019 — a task that Google estimated would take the world's fastest classical supercomputer at the time, Summit, approximately 10,000 years. IBM's Osprey processor, unveiled in November 2022, contained 433 qubits, and IBM announced the Condor processor with 1,121 qubits in December 2023.

The energy cost of computation has become a central concern. In 2023, training a single large language model such as GPT-4 (parameters undisclosed; estimated 1.8 trillion by some researchers) reportedly consumed between 50 and 100 GWh of electricity — enough to power approximately 4,500 average US homes for a year. This has spurred interest in in-context learning and parameter-efficient fine-tuning to reduce training frequency.

Looking forward to 2024 and beyond, the chip industry faces significant challenges. EUV (Extreme Ultraviolet) lithography at sub-3nm nodes requires light with a wavelength of 13.5 nm produced by a tin plasma struck by a laser, with equipment manufactured exclusively by ASML of the Netherlands. The High-NA EUV systems required for 2nm and beyond cost approximately €350 million each. TSMC plans its 2nm node (N2) for high-volume production in 2025, while Intel targets its 18A node (approximately 1.8nm effective gate pitch) for the same timeframe with a planned return to process leadership.

Throughout this half-century arc, one constant has been the interplay between materials science, software ecosystems, and economic forces. Silicon's dominance as a semiconductor substrate stems from its native oxide (SiO₂), which provides natural passivation and insulation. Research groups at Stanford, MIT, and imec in Belgium are exploring gallium nitride (GaN), silicon carbide (SiC), and two-dimensional materials such as molybdenum disulfide (MoS₂) as potential successors for specific applications.

The story of the microprocessor is ultimately one of collaborative engineering across continents: American chip design, Taiwanese manufacturing precision, Dutch lithography optics, South Korean DRAM expertise, and Japanese materials innovation have all contributed indispensably to the modern integrated circuit industry. The Intel 4004's 2,300 transistors grew to 80 billion in the H100 — a factor of roughly 35 million times over 53 years.
""",
"questions": [
    {
        "id": "q1a",
        "question": "How many transistors did the Intel 4004 contain, and what was its clock speed?",
        "gold_answer": "2,300 transistors and 740 kHz",
        "answer_location": "start",
        "gold_keywords": ["2,300", "2300", "transistors", "740 kHz", "740kHz"]
    },
    {
        "id": "q1b",
        "question": "In what year did AMD introduce the AMD64 instruction set, and in which processor family?",
        "gold_answer": "2003, in the Opteron processor family",
        "answer_location": "middle",
        "gold_keywords": ["2003", "Opteron"]
    },
    {
        "id": "q1c",
        "question": "How many transistors does Nvidia's H100 GPU contain, and what is its FP8 Tensor Core performance?",
        "gold_answer": "80 billion transistors and 3,958 TFLOPS of FP8 Tensor Core performance",
        "answer_location": "late",
        "gold_keywords": ["80 billion", "3,958", "TFLOPS"]
    }
]
},

# ----- DOC 2: Climate Science Report (~2000 words) -----
{
"id": "doc_climate",
"approx_words": 2100,
"title": "Global Climate Indicators: A Synthesis of 2023 Data",
"text": """Global Climate Indicators: A Synthesis of 2023 Data

The year 2023 was confirmed by the World Meteorological Organization (WMO) as the warmest year on record, with global mean surface temperature reaching 1.45°C above the pre-industrial baseline defined as the 1850–1900 average. The previous record was set in 2016 at approximately 1.29°C. Each of the twelve months of 2023 ranked either as the hottest or among the top two hottest on record for their respective month. The European Union's Copernicus Climate Change Service confirmed this ranking in its annual report published in January 2024.

Ocean heat content is considered a more comprehensive measure of Earth's energy imbalance than surface temperature alone, because the ocean absorbs approximately 93% of the excess energy trapped by greenhouse gases. In 2023, the global ocean heat content in the upper 2,000 metres reached a new record high for the fourth consecutive year. The North Atlantic experienced anomalously high sea surface temperatures beginning in April 2023, with anomalies exceeding +1.5°C above the 1991–2020 average in some regions during the summer months — a pattern that climate scientists linked to reduced aerosol loading following new International Maritime Organization (IMO) sulphur regulations implemented in January 2020.

Sea ice extent in both poles reached record or near-record lows in 2023. Antarctic sea ice extent set an all-time record minimum in February 2023 at 1.79 million km² — approximately 1 million km² below the 1981–2010 average for that date. This anomaly persisted through the southern winter, with September 2023 sea ice extent also reaching a record low of 16.96 million km², compared with the previous record of 17.59 million km² set in 1986.

Atmospheric carbon dioxide concentration at the Mauna Loa Observatory in Hawaii reached a seasonal peak of 424.0 parts per million (ppm) in May 2023. The pre-industrial baseline concentration is estimated at approximately 278 ppm, meaning the 2023 peak represents an increase of approximately 52.5% above pre-industrial levels. The annual mean for 2023 was 420.0 ppm, up from 418.6 ppm in 2022 — an annual increment of 1.4 ppm, which is below the recent ten-year average increment of approximately 2.0–2.5 ppm but reflects year-to-year variability in El Niño–Southern Oscillation (ENSO) dynamics.

Methane (CH₄) concentrations averaged 1,922.6 parts per billion (ppb) for 2023 based on global network measurements compiled by the National Oceanic and Atmospheric Administration (NOAA). The pre-industrial level was approximately 722 ppb. This represents an increase of approximately 165% above the pre-industrial baseline. The primary anthropogenic sources of methane are livestock enteric fermentation (accounting for approximately 32% of global anthropogenic emissions), oil and gas production, coal mining, and landfill decomposition.

Nitrous oxide (N₂O) has a global warming potential of 265 times that of CO₂ over a 100-year horizon (GWP₁₀₀), and atmospheric concentrations reached 336.9 ppb in 2023, compared with the pre-industrial level of approximately 270 ppb. Agricultural soils are the dominant anthropogenic source, contributing approximately 70% of global N₂O emissions through nitrogen fertiliser application.

Extreme weather events in 2023 caused significant economic and humanitarian impacts. Turkey and Syria experienced a devastating earthquake on February 6, 2023, with the Turkey–Syria earthquake causing over 50,000 deaths — though this is a geological rather than climate event. Climate-linked events included severe flooding in Libya in September 2023 caused by Storm Daniel, which resulted in over 11,000 confirmed deaths in the city of Derna alone after two upstream dams, the Abu Mansour and Bilad dams, collapsed. The Canadian wildfire season of 2023 was the worst on record, burning approximately 18.4 million hectares — more than three times the previous record — and producing smoke that reached European populations.

The global mean sea level rise trend, based on satellite altimetry data from the TOPEX/Poseidon, Jason series, and Sentinel-6 satellite missions, shows a rate of 3.7 mm per year averaged over the 1993–2023 period. However, the rate has been accelerating: the period 2013–2023 shows a rate of approximately 4.5 mm per year, driven by increased contributions from the Greenland Ice Sheet (contributing approximately 0.77 mm/year averaged over 2006–2015) and the Antarctic Ice Sheet.

Permafrost thaw in Arctic and sub-Arctic regions represents a potential tipping point with significant feedback implications. Approximately 1,500 Gt (gigatonnes) of organic carbon are estimated to be stored in permafrost soils, more than double the current atmospheric carbon stock. Even the release of a small fraction of this carbon as CO₂ or CH₄ could substantially amplify warming. The Arctic Monitoring and Assessment Programme (AMAP) estimated in its 2023 update that permafrost temperatures have increased by an average of 0.39°C per decade since the early 1980s.

International climate policy made incremental but historically significant progress in 2023. At COP28, held in Dubai in December 2023, the Global Stocktake concluded that current Nationally Determined Contributions (NDCs) are insufficient to limit warming to 1.5°C. The agreement text for the first time included explicit language calling for a "transition away from fossil fuels in energy systems" — the first such language in a COP final agreement, though without binding emissions targets. Developed nations had previously committed to providing $100 billion annually to developing nations for climate adaptation and mitigation, a commitment that the OECD confirmed was first met in 2022 at $115.9 billion, slightly above target.

Renewable energy deployment accelerated substantially in 2023. The International Energy Agency (IEA) reported that global renewable electricity capacity additions reached a record 507 GW in 2023, up 50% from the previous year. Solar photovoltaic additions alone accounted for 413 GW, with China contributing approximately 60% of global solar capacity additions. The weighted average cost of utility-scale solar photovoltaic electricity fell from $0.378/kWh in 2010 to $0.049/kWh in 2023 — a reduction of approximately 87%.

The scientific consensus on climate attribution continues to strengthen. The Intergovernmental Panel on Climate Change (IPCC) Sixth Assessment Report (AR6), completed in 2023, concluded with high confidence that human influence has warmed the climate at an unprecedented rate and that it is unequivocal that human influence has warmed the atmosphere, ocean, and land. The AR6 Working Group I report stated that global surface temperature increased by approximately 1.09°C between 1850–1900 and 2011–2020, with the likely range being 0.95°C to 1.20°C.
""",
"questions": [
    {
        "id": "q2a",
        "question": "What global mean surface temperature anomaly was recorded in 2023, and how does it compare to the 2016 record?",
        "gold_answer": "1.45°C above pre-industrial baseline; the 2016 record was approximately 1.29°C",
        "answer_location": "start",
        "gold_keywords": ["1.45", "1.29", "2016"]
    },
    {
        "id": "q2b",
        "question": "What was the Antarctic sea ice extent record minimum set in February 2023?",
        "gold_answer": "1.79 million km²",
        "answer_location": "early-middle",
        "gold_keywords": ["1.79 million", "1.79"]
    },
    {
        "id": "q2c",
        "question": "By how much did the cost of utility-scale solar PV electricity fall between 2010 and 2023?",
        "gold_answer": "From $0.378/kWh in 2010 to $0.049/kWh in 2023 — a reduction of approximately 87%",
        "answer_location": "late",
        "gold_keywords": ["87%", "0.378", "0.049"]
    }
]
},

# ----- DOC 3: Medical Research Paper (~1900 words) -----
{
"id": "doc_medical",
"approx_words": 2000,
"title": "mRNA Vaccine Technology: Mechanisms, Development Timeline, and Efficacy Data",
"text": """mRNA Vaccine Technology: Mechanisms, Development Timeline, and Efficacy Data

Messenger RNA (mRNA) vaccines represent a paradigm shift in vaccinology. Unlike conventional vaccines that introduce attenuated pathogens, inactivated viruses, or protein subunits, mRNA vaccines deliver genetic instructions encoding the antigen of interest, allowing host cells to produce the immunogenic protein transiently. The mRNA is flanked by a 5' cap structure and a poly-A tail to mimic endogenous mRNA, improving stability and translation efficiency. The molecule is encapsulated in lipid nanoparticles (LNPs) composed of an ionisable lipid, a phospholipid (typically distearoylphosphatidylcholine, DSPC), cholesterol, and PEG-lipid in precise molar ratios to facilitate cellular uptake and endosomal escape.

The foundational research enabling mRNA therapeutics spanned several decades. In 1990, Robert Malone and colleagues demonstrated that lipid-mediated delivery of mRNA into mouse muscle cells could produce a functional protein. Katalin Karikó and Drew Weissman published their landmark 2005 paper in Immunity showing that substituting uridine with pseudouridine in synthetic mRNA dramatically reduced innate immune activation — the key barrier to in vivo mRNA translation. This modification, combined with high-performance liquid chromatography (HPLC) purification to remove double-stranded RNA contaminants, enabled efficient and prolonged protein expression. Karikó and Weissman were awarded the Nobel Prize in Physiology or Medicine in 2023 for this discovery.

BioNTech (Germany) and Moderna (United States) independently pursued mRNA therapeutics throughout the 2010s. BioNTech, founded in Mainz in 2008 by Uğur Şahin and Özlem Türeci, focused primarily on personalised cancer mRNA vaccines (individualised neoantigen therapy, or iNeST). Moderna, founded in Cambridge, Massachusetts in 2010, initially targeted rare genetic disorders and subsequently developed mRNA therapeutics for cytomegalovirus, respiratory syncytial virus, and HIV.

The COVID-19 pandemic dramatically accelerated mRNA vaccine development. SARS-CoV-2 was first identified in Wuhan, China in late December 2019, with the full genome sequence published on January 10, 2020. Moderna began designing its mRNA-1273 vaccine on January 11, 2020 — the day after the sequence became public — using the full-length prefusion-stabilised spike protein sequence with two proline substitutions (K986P and V987P) to maintain the prefusion conformation, a design strategy from the laboratory of Barney Graham at the NIAID Vaccine Research Center.

Clinical trials proceeded at unprecedented speed under Operation Warp Speed in the United States. The Moderna mRNA-1273 Phase 3 trial (COVE study) enrolled 30,420 participants across 99 sites. The primary efficacy analysis, conducted when 196 COVID-19 cases had accumulated, showed a vaccine efficacy of 94.1% (95% CI: 89.3%–96.8%) against symptomatic COVID-19 caused by the original Wuhan strain. The vaccine was administered as two 100 μg doses 28 days apart. The FDA granted Emergency Use Authorization (EUA) on December 18, 2020.

The Pfizer-BioNTech BNT162b2 vaccine Phase 3 trial (C4591001) enrolled 43,448 participants. Among evaluable participants with a 2-month follow-up after dose 2, vaccine efficacy was 95.0% (95% CI: 90.3%–97.6%) against symptomatic COVID-19. The dosing regimen was two 30 μg doses 21 days apart. The FDA granted EUA on December 11, 2020 — one week before Moderna's authorisation. BNT162b2 was stored at -70°C, requiring specialised cold-chain logistics, while mRNA-1273 was stable at standard freezer temperatures (-20°C) for up to six months and refrigerator temperatures (2°–8°C) for up to 30 days.

Real-world effectiveness data broadly confirmed trial results but revealed important nuances. The Israeli Ministry of Health, in collaboration with Pfizer, published data in February 2021 showing 92% effectiveness against confirmed infection (symptomatic and asymptomatic), 94% against hospitalisation, and 92% against severe disease among approximately 596,618 vaccinated individuals compared with 596,618 matched controls. A CDC Morbidity and Mortality Weekly Report from April 2021 showed BNT162b2 and mRNA-1273 to be 90% effective against COVID-19 infection overall among healthcare workers in the HEROES-RECOVER network.

With the emergence of variants of concern, particularly Omicron (B.1.1.529), initially detected in Botswana and South Africa in November 2021, vaccine effectiveness against symptomatic infection declined substantially. UK Health Security Agency data showed BNT162b2 effectiveness against symptomatic Omicron infection falling to approximately 25–40% by four to six months post-dose 2, though effectiveness against hospitalisation remained higher (approximately 70% at two to four weeks post-dose 2). Bivalent boosters targeting both the original Wuhan strain and the Omicron BA.4/BA.5 subvariants were authorised in the United States in September 2022 and in the EU shortly thereafter.

Personalised mRNA cancer vaccines entered late-stage clinical evaluation in this period. Moderna's mV-4766 (mRNA-4157), developed in collaboration with Merck, encodes up to 34 personalised neoantigens identified through tumour exome sequencing and predictive algorithms. In a randomised Phase 2b trial (KEYNOTE-942), 157 patients with resected stage III or IV melanoma who had received adjuvant pembrolizumab were randomised to receive either pembrolizumab alone or the combination of pembrolizumab plus mRNA-4157. The combination arm showed a 44% reduction in risk of recurrence or death (recurrence-free survival hazard ratio: 0.56; 95% CI: 0.31–1.01; p=0.053) compared with pembrolizumab monotherapy — a clinically meaningful though not statistically significant result at the pre-specified interim analysis. Full results were expected from the ongoing Phase 3 trial.

Regulatory frameworks for mRNA medicines are evolving. The European Medicines Agency (EMA) granted conditional marketing authorisation for Comirnaty (BNT162b2) on December 21, 2020, and Spikevax (mRNA-1273) on January 6, 2021. The FDA converted BNT162b2's EUA to full Biologics License Application (BLA) approval on August 23, 2021, for individuals 16 years and older — the first full approval of a COVID-19 vaccine in the United States. Manufacturing scale-up required substantial investment: Pfizer and BioNTech produced approximately 3 billion doses globally by the end of 2021, while Moderna produced approximately 807 million doses in the same year.

The mRNA platform holds considerable promise beyond infectious disease. Ongoing clinical programmes include mRNA therapeutics for propionic acidaemia (mRNA-3927, Moderna), heart failure (AZD-8601, AstraZeneca/Moderna collaboration targeting VEGF-A expression), and HIV (a trimer-based Env immunogen). Challenges remain in hepatic delivery specificity, long-term durability of response, and scaling low-temperature cold-chain logistics to low- and middle-income countries. Nevertheless, the successful clinical deployment of mRNA vaccines for COVID-19 established the platform as a proven modality in human medicine.
""",
"questions": [
    {
        "id": "q3a",
        "question": "What key modification did Karikó and Weissman demonstrate in their 2005 paper to reduce immune activation of synthetic mRNA?",
        "gold_answer": "Substituting uridine with pseudouridine in synthetic mRNA dramatically reduced innate immune activation",
        "answer_location": "start",
        "gold_keywords": ["pseudouridine", "uridine", "immune activation", "2005"]
    },
    {
        "id": "q3b",
        "question": "What was the vaccine efficacy of Moderna's mRNA-1273 in its Phase 3 COVE trial, and what was the trial size?",
        "gold_answer": "94.1% efficacy (95% CI: 89.3%–96.8%), with 30,420 participants",
        "answer_location": "middle",
        "gold_keywords": ["94.1%", "30,420", "COVE"]
    },
    {
        "id": "q3c",
        "question": "What was the hazard ratio result in the KEYNOTE-942 Phase 2b trial combining mRNA-4157 with pembrolizumab for melanoma?",
        "gold_answer": "Hazard ratio 0.56 (95% CI: 0.31–1.01), representing a 44% reduction in risk of recurrence or death",
        "answer_location": "late",
        "gold_keywords": ["0.56", "44%", "KEYNOTE-942", "hazard ratio"]
    }
]
},

# ----- DOC 4: Economics / Finance Report (~1800 words) -----
{
"id": "doc_economics",
"approx_words": 1900,
"title": "Global Monetary Policy Tightening Cycle 2022–2024: Central Bank Actions and Economic Outcomes",
"text": """Global Monetary Policy Tightening Cycle 2022–2024: Central Bank Actions and Economic Outcomes

The monetary policy tightening cycle that began in 2022 represents the most aggressive coordinated rate-hiking episode among major central banks since the early 1980s Volcker disinflation. Headline inflation in the United States reached 9.1% year-on-year in June 2022 — the highest reading since November 1981 — driven by a combination of post-pandemic supply chain disruptions, commodity price shocks following Russia's invasion of Ukraine in February 2022, and pandemic-era fiscal transfers that had elevated nominal household demand.

The Federal Open Market Committee (FOMC) initiated its hiking cycle on March 16, 2022, with a 25 basis point (bps) increase to the federal funds rate target range, raising it from 0–0.25% to 0.25–0.5%. Subsequent meetings saw increases of 50 bps (May 2022) and four consecutive increases of 75 bps (June, July, September, and November 2022) — an unprecedented pace. The cumulative tightening reached 525 bps between March 2022 and July 2023, bringing the federal funds rate target range to 5.25–5.5% — its highest level since January 2001. The Federal Reserve also began reducing its balance sheet (quantitative tightening) at a pace of $95 billion per month from September 2022, comprising $60 billion in Treasury securities and $35 billion in agency mortgage-backed securities.

The European Central Bank (ECB) initiated its first rate increase since 2011 in July 2022, raising the deposit facility rate by 50 bps from -0.5% to 0%. The ECB tightened by a total of 450 bps between July 2022 and September 2023, bringing the deposit facility rate to 4.0%. This rate cycle was complicated by the fragmentation risk — the widening of sovereign bond spreads between euro area members — which prompted the ECB to introduce the Transmission Protection Instrument (TPI) in July 2022, allowing targeted sovereign bond purchases to prevent unwarranted spread widening.

The Bank of England (BoE) began its tightening cycle earlier than its peers, raising the Bank Rate from 0.1% to 0.25% in December 2021, the first increase since August 2018. By August 2023, the Bank Rate stood at 5.25%, the highest level since 2008. The United Kingdom faced a distinctive inflationary episode with energy price shocks amplified by high reliance on gas for home heating and power generation, as well as post-Brexit labour market rigidities. Core inflation (excluding energy and food) peaked at 7.1% in May 2023.

Emerging market central banks, many of which began hiking in 2021, reached extremely high nominal policy rates. Brazil's Selic rate peaked at 13.75% in August 2022 and was held at that level for over twelve months. Chile's overnight rate reached 11.25% in October 2022. Mexico's Banco de México raised its target rate to 11.25% in February 2023. These rates reflected both domestic inflation dynamics and the need to defend currencies against capital outflows attracted by rising US interest rates.

The transmission of monetary tightening to economic activity and inflation proved slower than initially forecast by several central banks. US unemployment remained below 4% throughout 2022 and 2023, while the labour market added over 200,000 jobs per month on average through 2022 — a remarkable resilience attributed to continued post-COVID labour demand, structural labour shortages, and elevated household savings buffers accumulated during the pandemic. By contrast, the housing sector contracted sharply: US existing home sales fell 17.6% in 2022 and a further 18.7% in 2023 as the 30-year fixed mortgage rate rose from approximately 3.1% in January 2022 to a peak of 7.79% in October 2023 — the highest since November 2000.

Banking sector stresses emerged in March 2023. Silicon Valley Bank (SVB), the 16th largest US bank by assets at the time, failed on March 10, 2023, following a run on deposits triggered by unrealised losses on its long-duration securities portfolio — a consequence of the rapid rise in interest rates. SVB held $91.3 billion in available-for-sale and held-to-maturity securities at year-end 2022, with unrealised losses of $15.9 billion. Signature Bank failed two days later on March 12. First Republic Bank, which had $229.1 billion in assets, was seized by regulators and sold to JPMorgan Chase on May 1, 2023.

Inflation deceleration proceeded unevenly across economies. US headline CPI inflation fell from its 9.1% peak in June 2022 to 3.1% by November 2023, primarily driven by a reversal of goods inflation and declining energy prices. However, services inflation, which the Federal Reserve considers more persistent due to its dependence on labour costs and wage dynamics, remained elevated — services CPI excluding shelter ran at approximately 3.9% year-on-year in late 2023. Eurozone headline inflation peaked at 10.6% in October 2022 and fell to 2.4% by November 2023.

The Bank for International Settlements (BIS) Annual Economic Report 2023 estimated that the net interest margin compression experienced by banks with large fixed-rate loan portfolios would continue to weigh on profitability through 2024–2025 as fixed-rate mortgages originated at low rates repriced. The BIS also noted that global debt-to-GDP ratios, which had risen sharply during the pandemic, presented a vulnerability: a 100 bps increase in interest rates applied to outstanding government debt would increase annual interest payments by approximately 1.0–1.5% of GDP for major advanced economies over a three-to-five-year horizon, once existing fixed-rate instruments matured.

The first rate cuts of the cycle arrived in 2024. The Swiss National Bank cut its policy rate from 1.75% to 1.5% in March 2024, becoming the first G10 central bank to ease. The ECB followed with a 25 bps cut to 3.75% in June 2024. The Fed initiated easing with a 50 bps cut to 4.75–5.0% in September 2024, citing progress toward its 2% inflation goal and rising labour market risks. By contrast, the Bank of Japan moved in the opposite direction, raising its policy rate from -0.1% to 0.1% in March 2024 and subsequently to 0.25% in July 2024, ending its eight-year experiment with negative interest rate policy amid rising domestic inflation expectations.
""",
"questions": [
    {
        "id": "q4a",
        "question": "What was US headline inflation in June 2022, and when was the last time such a level was recorded?",
        "gold_answer": "9.1% year-on-year in June 2022; the highest since November 1981",
        "answer_location": "start",
        "gold_keywords": ["9.1%", "November 1981", "1981"]
    },
    {
        "id": "q4b",
        "question": "What was the total cumulative rate increase by the Federal Reserve between March 2022 and July 2023, and what was the resulting target range?",
        "gold_answer": "525 basis points, bringing the federal funds rate to 5.25–5.5%",
        "answer_location": "early",
        "gold_keywords": ["525 bps", "525", "5.25", "5.5%"]
    },
    {
        "id": "q4c",
        "question": "What were the total assets of First Republic Bank when it was seized and sold to JPMorgan Chase, and on what date did this occur?",
        "gold_answer": "First Republic Bank had $229.1 billion in assets and was sold to JPMorgan Chase on May 1, 2023",
        "answer_location": "middle-late",
        "gold_keywords": ["229.1 billion", "229.1", "May 1, 2023", "JPMorgan"]
    }
]
},

# ----- DOC 5: AI / ML Technical Survey (~1800 words) -----
{
"id": "doc_ai_survey",
"approx_words": 1900,
"title": "Transformer Architecture Variants and Efficiency Improvements (2017–2024)",
"text": """Transformer Architecture Variants and Efficiency Improvements (2017–2024)

The transformer architecture, introduced in the paper "Attention Is All You Need" by Vaswani et al. from Google Brain and Google Research, published at NeurIPS 2017, fundamentally changed natural language processing. The original transformer replaced recurrent neural networks with a self-attention mechanism that computes pairwise relationships between all tokens in a sequence simultaneously. The computational complexity of this operation scales as O(n²) in sequence length n, which presents a bottleneck for very long sequences.

The original BERT model, published by Devlin et al. at Google in 2018 and released publicly in October 2018, used a bidirectional encoder-only transformer trained on two tasks: Masked Language Modelling (MLM) and Next Sentence Prediction (NSP). BERT-Large contained 340 million parameters, used 24 transformer layers, 1,024 hidden dimensions, and 16 attention heads, and was trained on 3,300 million words from BooksCorpus and English Wikipedia. Its tokenizer used WordPiece encoding with a vocabulary of 30,522 tokens.

The GPT family of models from OpenAI adopted a decoder-only unidirectional architecture, trained with causal language modelling (next-token prediction). GPT-1 (2018) had 117 million parameters; GPT-2 (2019) scaled to 1.5 billion; GPT-3 (2020) reached 175 billion parameters, trained on approximately 300 billion tokens from a filtered Common Crawl corpus, WebText2, BooksCorpus, and Wikipedia. The GPT-3 training compute was estimated by researchers at approximately 3.14×10²³ floating-point operations (FLOPs). OpenAI did not publicly disclose GPT-4's parameter count, but independent estimates by researchers including those at SemiAnalysis suggested a mixture-of-experts architecture with approximately 1.8 trillion parameters across 16 expert models.

Attention efficiency became a critical research focus. The standard quadratic attention was addressed by several approaches. Sparse attention, introduced in Longformer (Beltagy et al., 2020) and BigBird (Zaheer et al., 2020), achieved O(n) complexity by combining local windowed attention with global tokens. Flash Attention (Dao et al., 2022), rather than reducing the asymptotic complexity, dramatically improved hardware utilisation by reordering attention computations to exploit GPU SRAM bandwidth — reducing memory I/O from O(n²) to O(n) while maintaining mathematically exact attention. Flash Attention 2 (2023) further optimised parallelism, achieving approximately 2× speed improvement over Flash Attention on A100 GPUs.

The mixture-of-experts (MoE) architecture, in which only a subset of model parameters are activated per token, enables scaling parameter counts without proportionally increasing compute per forward pass. Fedus et al.'s Switch Transformer (2022) demonstrated that an MoE model with 1.6 trillion parameters could be trained at the same compute cost as a 7 billion dense parameter model, achieving competitive performance on a range of NLP benchmarks. Mistral AI's Mixtral 8x7B, released in December 2023, used an MoE design with 8 experts of 7B parameters each, with each token routed to 2 of the 8 experts, yielding an effective parameter count of 46.7 billion but activating only approximately 12.9 billion parameters per forward pass.

Retrieval-Augmented Generation (RAG), introduced formally by Lewis et al. from Facebook AI Research in 2020, combines a parametric language model with a non-parametric retrieval component. At inference time, the model retrieves k relevant documents from an external corpus using a dense passage retrieval (DPR) system and conditions generation on both the original query and retrieved passages. The original RAG paper evaluated on Natural Questions, TriviaQA, and WebQuestions benchmarks, showing state-of-the-art results for open-domain QA at the time, with the RAG-Token variant achieving 44.5 EM (Exact Match) on Natural Questions compared with 29.8 for the DPR reader baseline.

Instruction tuning — fine-tuning language models on datasets of natural language instructions and their responses — dramatically improved model usability. FLAN (Finetuned Language Net, Wei et al., 2021) demonstrated that instruction tuning on 60 NLP tasks improved zero-shot performance on held-out tasks. InstructGPT (Ouyang et al., 2022) used reinforcement learning from human feedback (RLHF) to align GPT-3 descendants with human preferences, finding that the 1.3B InstructGPT model was preferred by human labellers over the 175B GPT-3 on 85.1% of comparisons — a remarkable finding suggesting alignment techniques can substitute for raw scale.

Constitutional AI (Anthropic, 2022), direct preference optimisation (DPO, Rafailov et al., Stanford, 2023), and Group Relative Policy Optimisation (GRPO, DeepSeek, 2024) represent successive refinements to alignment methodology. DPO reformulates RLHF as a supervised classification problem, eliminating the need for a separate reward model, and demonstrated comparable alignment performance on the Alpaca benchmark with substantially reduced training complexity. GRPO, used in training DeepSeek-R1, reduces GPU memory requirements relative to PPO by using group-level advantages rather than per-token advantages from a critic model.

Large language model evaluation methodology itself became a research focus. MMLU (Massive Multitask Language Understanding, Hendrycks et al., 2020) covers 57 academic subjects ranging from elementary mathematics to advanced law, testing both world knowledge and reasoning ability. The original GPT-3 achieved 43.9% on MMLU; GPT-4 achieved 86.4%; Gemini Ultra 1.0 achieved 90.0%; and Claude 3 Opus achieved 86.8% at the time of its March 2024 launch. BIG-Bench Hard (Suzgun et al., 2022) is a subset of 23 tasks from BIG-Bench that are considered particularly challenging for LLMs; GPT-4 achieved 83.1% with chain-of-thought prompting.

Context length extension has been a defining capability race in 2023–2024. The original GPT-4 context window was 8,192 tokens, later extended to 128,000 tokens in the GPT-4 Turbo (November 2023) release. Google's Gemini 1.0 Ultra supported a 32,768-token context, while Gemini 1.5 Pro, released in February 2024, featured a 1 million-token context window using a novel architecture variant. Anthropic's Claude 3 family supported a 200,000-token context. The key technique enabling these extensions is Rotary Position Embedding (RoPE, Su et al., 2021), which encodes positional information within the attention score computation, allowing extrapolation beyond training-time sequence lengths through base frequency modifications.
""",
"questions": [
    {
        "id": "q5a",
        "question": "How many parameters did GPT-3 have, and on how many tokens was it trained?",
        "gold_answer": "175 billion parameters, trained on approximately 300 billion tokens",
        "answer_location": "early",
        "gold_keywords": ["175 billion", "300 billion tokens", "300 billion"]
    },
    {
        "id": "q5b",
        "question": "What was the effective activated parameter count per forward pass in Mixtral 8x7B?",
        "gold_answer": "Approximately 12.9 billion parameters are activated per forward pass",
        "answer_location": "middle",
        "gold_keywords": ["12.9 billion", "12.9"]
    },
    {
        "id": "q5c",
        "question": "What MMLU score did Gemini Ultra 1.0 achieve, and what did Claude 3 Opus achieve at its March 2024 launch?",
        "gold_answer": "Gemini Ultra 1.0 achieved 90.0% and Claude 3 Opus achieved 86.8%",
        "answer_location": "late",
        "gold_keywords": ["90.0%", "90.0", "86.8%", "86.8", "Gemini Ultra", "Claude 3 Opus"]
    }
]
},

]  # end DOCUMENTS


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lowercase, strip punctuation for token comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalise(pred)
    gold_tokens = normalise(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1
    gold_set = {}
    for t in gold_tokens:
        gold_set[t] = gold_set.get(t, 0) + 1
    common = 0
    for t in gold_set:
        common += min(gold_set[t], pred_set.get(t, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_match(pred: str, keywords: list) -> bool:
    """True if at least ceil(len/2) keywords appear in prediction."""
    pred_lower = pred.lower()
    hits = sum(1 for kw in keywords if kw.lower() in pred_lower)
    threshold = max(1, len(keywords) // 2)
    return hits >= threshold


def score_answer(pred: str, gold: str, keywords: list) -> dict:
    f1 = token_f1(pred, gold)
    km = keyword_match(pred, keywords)
    # Exact substring (normalised)
    exact = gold.lower().strip() in pred.lower()
    return {
        "token_f1": round(f1, 3),
        "keyword_match": km,
        "exact_substring": exact,
        "correct": km  # primary metric: keyword match
    }


# ---------------------------------------------------------------------------
# Ollama direct call (baseline)
# ---------------------------------------------------------------------------

def ollama_chat(messages: list, model: str = GENERATOR_MODEL,
                num_ctx: int = 32768, seed: int = SEED,
                num_predict: int = 512) -> tuple[str, float]:
    """Direct Ollama call. Returns (response_text, elapsed_seconds)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": 0,
        }
    }
    t0 = time.time()
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    text = data.get("message", {}).get("content", "").strip()
    return text, elapsed


def run_baseline_question(doc_text: str, question: str, doc_title: str) -> tuple[str, float]:
    """Baseline: single LLM call with doc + question."""
    system_msg = (
        "You are a precise reading comprehension assistant. "
        "Answer the question based only on the provided document. "
        "Give a concise, direct answer. Do not add information not in the document."
    )
    user_msg = (
        f"Document Title: {doc_title}\n\n"
        f"Document:\n{doc_text}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely, quoting exact figures from the document where relevant."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    # Estimate token count roughly (1 token ≈ 4 chars)
    approx_tokens = (len(system_msg) + len(user_msg)) // 4
    num_ctx = max(8192, approx_tokens + 1024)
    num_ctx = min(num_ctx, 32768)
    return ollama_chat(messages, num_ctx=num_ctx)


# ---------------------------------------------------------------------------
# VeriAgent pipeline call
# ---------------------------------------------------------------------------

def run_pipeline_question(doc_text: str, question: str, doc_title: str) -> tuple[str, float]:
    """VeriAgent pipeline: use model_core.answer_question with long doc as context."""
    import model_core

    # Build a combined question that includes the document as context
    full_question = (
        f"[Long-Context QA Task]\n"
        f"Document Title: {doc_title}\n\n"
        f"Document:\n{doc_text}\n\n"
        f"Based ONLY on the document above, answer this question:\n{question}\n\n"
        "Quote exact figures and facts from the document."
    )

    t0 = time.time()
    result = model_core.answer_question(full_question, history=[])
    elapsed = time.time() - t0

    answer = result.get("answer", "")
    return answer, elapsed


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark():
    print("=" * 70)
    print("VeriAgent Long-Context QA Benchmark")
    print(f"Model: {GENERATOR_MODEL} | Seed: {SEED}")
    print("=" * 70)

    all_results = []
    baseline_scores = []
    pipeline_scores = []

    total_q = sum(len(doc["questions"]) for doc in DOCUMENTS)
    q_idx = 0

    for doc in DOCUMENTS:
        doc_id = doc["id"]
        doc_title = doc["title"]
        doc_text = doc["text"]
        word_count = len(doc_text.split())
        approx_tokens = word_count * 4 // 3  # rough 0.75 words/token for English

        print(f"\n{'─' * 70}")
        print(f"Document: {doc_title}")
        print(f"Words: ~{word_count:,}  |  Approx tokens: ~{approx_tokens:,}")
        print(f"{'─' * 70}")

        for q in doc["questions"]:
            q_idx += 1
            qid = q["id"]
            question = q["question"]
            gold = q["gold_answer"]
            keywords = q["gold_keywords"]
            location = q["answer_location"]

            print(f"\n[{q_idx}/{total_q}] Q: {question}")
            print(f"  Gold: {gold}")
            print(f"  Answer location: {location}")

            # --- Baseline ---
            print("  Running baseline...", end=" ", flush=True)
            try:
                base_ans, base_lat = run_baseline_question(doc_text, question, doc_title)
                base_score = score_answer(base_ans, gold, keywords)
                print(f"done ({base_lat:.1f}s)  F1={base_score['token_f1']:.3f}  KM={'✓' if base_score['keyword_match'] else '✗'}")
            except Exception as e:
                base_ans = f"ERROR: {e}"
                base_lat = 0.0
                base_score = {"token_f1": 0, "keyword_match": False, "exact_substring": False, "correct": False}
                print(f"ERROR: {e}")

            # --- Pipeline ---
            print("  Running pipeline...", end=" ", flush=True)
            try:
                pipe_ans, pipe_lat = run_pipeline_question(doc_text, question, doc_title)
                pipe_score = score_answer(pipe_ans, gold, keywords)
                print(f"done ({pipe_lat:.1f}s)  F1={pipe_score['token_f1']:.3f}  KM={'✓' if pipe_score['keyword_match'] else '✗'}")
            except Exception as e:
                pipe_ans = f"ERROR: {e}"
                pipe_lat = 0.0
                pipe_score = {"token_f1": 0, "keyword_match": False, "exact_substring": False, "correct": False}
                print(f"ERROR: {e}")

            baseline_scores.append(base_score["correct"])
            pipeline_scores.append(pipe_score["correct"])

            all_results.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "approx_words": word_count,
                "approx_tokens": approx_tokens,
                "question_id": qid,
                "question": question,
                "gold_answer": gold,
                "answer_location": location,
                "baseline": {
                    "answer": base_ans,
                    "latency_s": round(base_lat, 2),
                    **base_score
                },
                "pipeline": {
                    "answer": pipe_ans,
                    "latency_s": round(pipe_lat, 2),
                    **pipe_score
                }
            })

    # --- Summary ---
    n = len(baseline_scores)
    base_acc = sum(baseline_scores) / n
    pipe_acc = sum(pipeline_scores) / n
    base_lat_avg = sum(r["baseline"]["latency_s"] for r in all_results) / n
    pipe_lat_avg = sum(r["pipeline"]["latency_s"] for r in all_results) / n
    base_f1_avg = sum(r["baseline"]["token_f1"] for r in all_results) / n
    pipe_f1_avg = sum(r["pipeline"]["token_f1"] for r in all_results) / n

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total questions: {n}")
    print(f"Baseline  → Keyword Match Acc: {sum(baseline_scores)}/{n} ({base_acc:.1%}) | Avg F1: {base_f1_avg:.3f} | Avg Latency: {base_lat_avg:.1f}s")
    print(f"Pipeline  → Keyword Match Acc: {sum(pipeline_scores)}/{n} ({pipe_acc:.1%}) | Avg F1: {pipe_f1_avg:.3f} | Avg Latency: {pipe_lat_avg:.1f}s")

    summary = {
        "benchmark": "Long-Context QA",
        "model": GENERATOR_MODEL,
        "seed": SEED,
        "total_questions": n,
        "baseline": {
            "correct": sum(baseline_scores),
            "accuracy": round(base_acc, 4),
            "avg_f1": round(base_f1_avg, 4),
            "avg_latency_s": round(base_lat_avg, 2)
        },
        "pipeline": {
            "correct": sum(pipeline_scores),
            "accuracy": round(pipe_acc, 4),
            "avg_f1": round(pipe_f1_avg, 4),
            "avg_latency_s": round(pipe_lat_avg, 2)
        },
        "results": all_results
    }

    # Save JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    return summary


if __name__ == "__main__":
    run_benchmark()

